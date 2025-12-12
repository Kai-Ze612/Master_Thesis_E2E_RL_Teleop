"""
Pure Python Teleoperation Simulation with ROS 2 Publishing.
- Runs Physics/AI Loop independently.
- Publishes State, EE Pose, Inference Time, AND DELAY METRICS.
- EE Pose and Delay Metrics at 50 Hz (reduced from control freq)
"""

import numpy as np
import torch
import time
import mujoco
import mujoco.viewer
from collections import deque
from dataclasses import dataclass, field
import threading

# ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

# Project Imports
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.inverse_kinematics import IKSolver
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator, Actor
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

# =============================================================================
# PUBLISH RATE CONFIGURATION
# =============================================================================
EE_POSE_PUBLISH_FREQ = 50  # Hz - for EE pose and delay metrics
METRICS_PUBLISH_FREQ = 50  # Hz - for delay metrics (obs/act delay)

# =============================================================================
# 1. LEADER ROBOT
# =============================================================================
@dataclass(frozen=True)
class TrajectoryParams:
    center: np.ndarray = field(default_factory=lambda: cfg.TRAJECTORY_CENTER.copy())
    scale: np.ndarray = field(default_factory=lambda: cfg.TRAJECTORY_SCALE.copy())
    frequency: float = cfg.TRAJECTORY_FREQUENCY
    initial_phase: float = 0.0

class Figure8Trajectory:
    def __init__(self, params: TrajectoryParams): self._params = params
    def compute_position(self, t: float) -> np.ndarray:
        phase = t * self._params.frequency * 2 * np.pi + self._params.initial_phase
        dx = self._params.scale[0] * np.sin(phase)
        dy = self._params.scale[1] * np.sin(phase / 2)
        dz = self._params.scale[2] * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz])

class LeaderRobot:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.ik_solver = IKSolver(self.model, cfg.JOINT_LIMITS_LOWER, cfg.JOINT_LIMITS_UPPER)
        self.ee_site_id = self.model.site(cfg.EE_BODY_NAME.replace("body", "site") if "body" in cfg.EE_BODY_NAME else "panda_ee_site").id
        
        self.q = cfg.INITIAL_JOINT_CONFIG.copy()
        self.qd = np.zeros(cfg.N_JOINTS)
        self.q_prev = self.q.copy()
        
        self.params = TrajectoryParams()
        self.generator = Figure8Trajectory(self.params)
        
        self.data.qpos[:cfg.N_JOINTS] = self.q
        mujoco.mj_forward(self.model, self.data)
        self.start_ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        self.traj_start_ee_pos = self.generator.compute_position(0.0)
        self.ik_solver.reset_trajectory(self.q)

    def step(self, t, dt):
        if t < cfg.WARM_UP_DURATION:
            alpha = t / cfg.WARM_UP_DURATION
            target_pos = (1 - alpha) * self.start_ee_pos + alpha * self.traj_start_ee_pos
        else:
            target_pos = self.generator.compute_position(t - cfg.WARM_UP_DURATION)

        q_target, success, _ = self.ik_solver.solve(target_pos, self.q)
        if not success or q_target is None: q_target = self.q.copy()

        self.qd = (q_target - self.q_prev) / dt
        self.q_prev = self.q.copy()
        self.q = q_target.copy()
        
        self.data.qpos[:cfg.N_JOINTS] = self.q
        mujoco.mj_kinematics(self.model, self.data)
        
        return self.q, self.qd

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

# =============================================================================
# 2. AI AGENT
# =============================================================================
class Agent:
    def __init__(self, device):
        self.device = device
        self.use_fp16 = (device.type == 'cuda')
        self.lstm = StateEstimator().to(device)
        self.actor = Actor(state_dim=cfg.OBS_DIM).to(device)
        self._load_models()
        if self.use_fp16:
            self.lstm.half()
            self.actor.half()
            
        self.leader_q_hist = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_hist = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.remote_q_hist = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        self.remote_qd_hist = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
       
        for _ in range(cfg.REMOTE_HISTORY_LEN):
            self.remote_q_hist.append(np.zeros(cfg.N_JOINTS))
            self.remote_qd_hist.append(np.zeros(cfg.N_JOINTS))
            
        self.hidden_state = None
        self.last_pred_norm = None
        self.ema_pred = None

    def _load_models(self):
        lstm_ckpt = torch.load(cfg.LSTM_MODEL_PATH, map_location=self.device, weights_only=False)
        self.lstm.load_state_dict(lstm_ckpt.get('state_estimator_state_dict', lstm_ckpt))
        self.lstm.eval()
        
        sac_ckpt = torch.load(cfg.RL_MODEL_PATH, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(sac_ckpt['actor_state_dict'])
        self.actor.eval()

    def _normalize(self, q, qd, delay):
        q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
        qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
        return np.concatenate([q_norm, qd_norm, [delay]])

    def _denormalize(self, pred):
        q = (pred[:7] * cfg.Q_STD) + cfg.Q_MEAN
        qd = (pred[7:] * cfg.QD_STD) + cfg.QD_MEAN
        return q, qd

    def inference(self, delayed_leader_packets, delay_scalar, rem_q, rem_qd):
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        if delayed_leader_packets:
            seq_data = []
            for pkt in delayed_leader_packets:
                self.leader_q_hist.append(pkt['q'])
                self.leader_qd_hist.append(pkt['qd'])
                seq_data.append(self._normalize(pkt['q'], pkt['qd'], delay_scalar))
            
            if len(self.leader_q_hist) >= cfg.RNN_SEQUENCE_LENGTH:
                seq_t = torch.tensor(np.array(seq_data), dtype=dtype, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    lstm_out, self.hidden_state = self.lstm.lstm(seq_t, self.hidden_state)
                    last_hidden = lstm_out[:, -1, :]
                    vel_pred = self.lstm.fc(last_hidden)
                    prev_state = seq_t[:, -1, :14]
                    self.last_pred_norm = prev_state + vel_pred * self.lstm.dt_scale

        elif self.last_pred_norm is not None:
            delay_t = torch.tensor([[[delay_scalar]]], dtype=dtype, device=self.device)
            state_in = self.last_pred_norm.unsqueeze(1)
            curr_in = torch.cat([state_in, delay_t], dim=2)
            with torch.no_grad():
                self.last_pred_norm, self.hidden_state = self.lstm.forward_step(curr_in, self.hidden_state)
                self.last_pred_norm = self.last_pred_norm.squeeze(1)

        if self.last_pred_norm is None:
            return rem_q, np.zeros(7), np.zeros(7)

        pred_np = self.last_pred_norm.float().cpu().numpy()[0]
        pred_q, pred_qd = self._denormalize(pred_np)
        
        if self.ema_pred is None: self.ema_pred = pred_q
        else: self.ema_pred = cfg.PREDICTION_EMA_ALPHA * pred_q + (1.0-cfg.PREDICTION_EMA_ALPHA)*self.ema_pred
        final_q = self.ema_pred

        self.remote_q_hist.append(rem_q)
        self.remote_qd_hist.append(rem_qd)
        
        obs = np.concatenate([
            rem_q, rem_qd,
            np.concatenate(list(self.remote_q_hist)),
            np.concatenate(list(self.remote_qd_hist)),
            final_q, pred_qd,
            (final_q - rem_q), (pred_qd - rem_qd),
            [delay_scalar]
        ]).astype(np.float32)
        
        obs_t = torch.tensor(obs, dtype=dtype, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.sample(obs_t, deterministic=True)[0]
        
        return final_q, pred_qd, action.float().cpu().numpy().flatten()

# =============================================================================
# 3. REMOTE ROBOT
# =============================================================================
class RemoteRobot:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = self.model.site(cfg.EE_BODY_NAME.replace("body", "site") if "body" in cfg.EE_BODY_NAME else "panda_ee_site").id
        
        self.data.qpos[:cfg.N_JOINTS] = cfg.INITIAL_JOINT_CONFIG
        mujoco.mj_step(self.model, self.data)

    def step(self, target_q, target_qd, tau_rl):
        q = self.data.qpos[:cfg.N_JOINTS].copy()
        qd = self.data.qvel[:cfg.N_JOINTS].copy()
       
        q_err = target_q - q
        qd_err = target_qd - qd
        acc_des = cfg.DEFAULT_KP_REMOTE * q_err + cfg.DEFAULT_KD_REMOTE * qd_err
        
        self.data.qpos[:cfg.N_JOINTS] = q
        self.data.qvel[:cfg.N_JOINTS] = qd
        self.data.qacc[:cfg.N_JOINTS] = acc_des
        mujoco.mj_inverse(self.model, self.data)
        tau_id = self.data.qfrc_inverse[:cfg.N_JOINTS].copy()
        
        tau = tau_id + tau_rl
        tau = np.clip(tau, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
        tau[-1] = 0.0
        print(f" tau: {tau}")
        self.data.ctrl[:cfg.N_JOINTS] = tau
        mujoco.mj_step(self.model, self.data)
        return q, qd

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

# =============================================================================
# 4. ROS 2 PUBLISHER (WITH RATE CONTROL)
# =============================================================================
class SimPublisher(Node):
    def __init__(self, control_freq: int):
        super().__init__('sim_publisher')
        
        # Full-rate publishers (joint states)
        self.pub_leader_q = self.create_publisher(JointState, 'leader/joint_states', 100)
        self.pub_remote_q = self.create_publisher(JointState, 'remote/joint_states', 100)
        self.pub_inf_time = self.create_publisher(Float32, 'agent/inference_time_ms', 100)
        
        # Reduced-rate publishers (EE pose and delay metrics)
        self.pub_leader_ee = self.create_publisher(PointStamped, 'leader/ee_pose', 50)
        self.pub_remote_ee = self.create_publisher(PointStamped, 'remote/ee_pose', 50)
        self.pub_obs_delay = self.create_publisher(Float32, 'agent/obs_delay_steps', 50)
        self.pub_act_delay = self.create_publisher(Float32, 'agent/act_delay_steps', 50)
        
        # Rate control
        self.control_freq = control_freq
        self.ee_skip = max(1, control_freq // EE_POSE_PUBLISH_FREQ)
        self.metrics_skip = max(1, control_freq // METRICS_PUBLISH_FREQ)
        self.step_count = 0
        
        self.get_logger().info(
            f"Publishing rates - Control: {control_freq} Hz, "
            f"EE Pose: {control_freq // self.ee_skip} Hz, "
            f"Delay Metrics: {control_freq // self.metrics_skip} Hz"
        )

    def publish_all(self, gt_q, rem_q, leader_ee, remote_ee, inf_ms, obs_delay, act_delay):
        now = self.get_clock().now().to_msg()
        self.step_count += 1
        
        # Helper functions
        def create_js(q):
            msg = JointState()
            msg.header.stamp = now
            msg.name = [f'panda_joint{i+1}' for i in range(7)]
            msg.position = q.tolist()
            return msg
            
        def create_ps(pos):
            msg = PointStamped()
            msg.header.stamp = now
            msg.header.frame_id = "world"
            msg.point.x, msg.point.y, msg.point.z = float(pos[0]), float(pos[1]), float(pos[2])
            return msg

        # --- FULL RATE: Joint states (control feedback) ---
        self.pub_leader_q.publish(create_js(gt_q))
        self.pub_remote_q.publish(create_js(rem_q))
        
        # --- FULL RATE: Inference time (diagnostic) ---
        t_msg = Float32()
        t_msg.data = float(inf_ms)
        self.pub_inf_time.publish(t_msg)
        
        # --- 50 Hz: EE Pose ---
        if self.step_count % self.ee_skip == 0:
            self.pub_leader_ee.publish(create_ps(leader_ee))
            self.pub_remote_ee.publish(create_ps(remote_ee))
        
        # --- 50 Hz: Delay Metrics ---
        if self.step_count % self.metrics_skip == 0:
            obs_msg = Float32()
            obs_msg.data = float(obs_delay)
            self.pub_obs_delay.publish(obs_msg)
            
            act_msg = Float32()
            act_msg.data = float(act_delay)
            self.pub_act_delay.publish(act_msg)

# =============================================================================
# 5. MAIN LOOP
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    
    control_freq = cfg.DEFAULT_CONTROL_FREQ
    publisher = SimPublisher(control_freq)
    
    leader = LeaderRobot()
    remote = RemoteRobot()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(device)
    
    dt = 1.0 / control_freq
    delay_sim = DelaySimulator(control_freq, ExperimentConfig.MEDIUM_DELAY, seed=42)
    
    leader_packet_queue = deque()
    action_delay_queue = deque()
    
    # Pre-fill history
    dummy_q = cfg.INITIAL_JOINT_CONFIG
    dummy_qd = np.zeros(7)
    for _ in range(cfg.RNN_SEQUENCE_LENGTH + 5):
        agent.leader_q_hist.append(dummy_q)
        agent.leader_qd_hist.append(dummy_qd)

    # =========================================================================
    # VISUALIZATION SETUP (MISSING IN ORIGINAL SNIPPET)
    # =========================================================================
    # We use deques with a maxlen to create a "sliding window" trail.
    # Adjust maxlen to control how long the tail remains visible.
    leader_target_trace = [] 
    remote_ee_trace = []
    # =========================================================================

    print(f"STARTING SIMULATION (ROS 2 ENABLED)")
    print(f"  Control Loop: {control_freq} Hz")
    
    sim_time = 0.0
    steps = 0
    
    with mujoco.viewer.launch_passive(remote.model, remote.data) as viewer:
        while viewer.is_running() and rclpy.ok():
            step_start = time.perf_counter()
            
            # 1. Leader Stepping
            gt_q, gt_qd = leader.step(sim_time, dt)
            leader_packet_queue.append({'q': gt_q, 'qd': gt_qd, 't': sim_time})
            leader_ee = leader.get_ee_pos()
            
            # 2. Network (Obs Delay)
            obs_delay_steps = delay_sim.get_observation_delay_steps(len(agent.leader_q_hist))
            delay_scalar = float(obs_delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
            received = []
            while leader_packet_queue:
                pkt = leader_packet_queue[0]
                if sim_time - pkt['t'] >= (obs_delay_steps * dt):
                    received.append(leader_packet_queue.popleft())
                else: break
            
            # 3. Inference
            curr_q = remote.data.qpos[:7].copy()
            curr_qd = remote.data.qvel[:7].copy()
            t0 = time.perf_counter()
            pred_q, pred_qd, tau_rl = agent.inference(received, delay_scalar, curr_q, curr_qd)
            inference_ms = (time.perf_counter() - t0) * 1000.0
            
            # 4. Action Delay
            action_delay_queue.append({'q': pred_q, 'qd': pred_qd, 'tau': tau_rl})
            act_delay_steps = delay_sim.get_action_delay_steps()
            if act_delay_steps >= len(action_delay_queue):
                cmd = {'q': cfg.INITIAL_JOINT_CONFIG, 'qd': np.zeros(7), 'tau': np.zeros(7)}
            else:
                cmd = action_delay_queue[-1 - act_delay_steps]
                
            # 5. Physics
            rem_q, rem_qd = remote.step(cmd['q'], cmd['qd'], cmd['tau'])
            remote_ee = remote.get_ee_pos()
            
            # 6. Publish ROS 2
            publisher.publish_all(gt_q, rem_q, leader_ee, remote_ee, inference_ms, obs_delay_steps, act_delay_steps)

            # =================================================================
            # 7. TRAJECTORY VISUALIZATION LOGIC
            # =================================================================
            
            # A. Update Buffers
            # We update less frequently (e.g., every 10 steps) to improve performance
            if steps % 10 == 0:
                # Calculate the ideal target position based on the generator
                traj_time = max(0, sim_time - cfg.WARM_UP_DURATION)
                leader_target_pos = leader.generator.compute_position(traj_time)
                
                leader_target_trace.append(leader_target_pos.copy())
                remote_ee_trace.append(remote_ee.copy())

            # B. Draw Geometries
            # Access the user scene to add dynamic geometries
            if viewer.user_scn:
                
                # 1. Reset geometry count for this frame
                viewer.user_scn.ngeom = 0
                
                # 2. Define geometry adding helper
                def add_geom(pos, size, rgba):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: return
                    
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[size, 0, 0],
                        pos=pos,
                        mat=np.eye(3).flatten(),
                        rgba=rgba
                    )
                    viewer.user_scn.ngeom += 1
                
                # 3. Draw Leader Trace (Red)
                for pos in leader_target_trace:
                    add_geom(pos, 0.01, [1, 0, 0, 0.5]) 
                    
                # 4. Draw Remote Trace (Blue)
                for pos in remote_ee_trace:
                    add_geom(pos, 0.01, [0, 0, 1, 0.5]) 

            # C. Sync & Logging
            steps += 1
            sim_time += dt
            viewer.sync() # Propagates the new geoms to the renderer
            
            elapsed = time.perf_counter() - step_start
            if elapsed < dt: time.sleep(dt - elapsed)
                
            if steps % 20 == 0:
                track_err = np.linalg.norm(rem_q - gt_q)
                print(f"[Step {steps}] Infer: {inference_ms:.2f}ms | Joint Err: {track_err:.4f}")

    publisher.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    # This block must NOT be indented
    main()