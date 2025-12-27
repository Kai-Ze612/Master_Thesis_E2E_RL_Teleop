import os
import sys
import time
import numpy as np
import torch
import mujoco
from collections import deque
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Header

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_agent_dir = os.path.abspath(os.path.join(current_dir, "../rl_agent_autoregressive"))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
sys.path.append(rl_agent_dir)

# --- IMPORTS ---
from SBSP_predictor import SBSPPredictor
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import Actor
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.inverse_kinematics import IKSolver
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

# --- CONFIGURATION ---
# UPDATE THESE PATHS IF NECESSARY
RL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent_autoregressive/rl_training_output/SBSP_SAC_MEDIUM_DELAY_figure_8_20251205_192034/best_policy.pth"
SBSP_MODEL = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent_autoregressive/models/FetchPush-RemotePDNorm-v0/2-256-1_step_prediction_sd.pt"
SBSP_STATS = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent_autoregressive/models/FetchPush-RemotePDNorm-v0/2-256-1_step_prediction_sd_params.pickle"

CONTROL_FREQ = 50 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# CLASSES
# =============================================================================
class TimeBasedQueue:
    def __init__(self):
        self.queue = deque()
        self.last_valid_data = None
    def send(self, data, current_time, delay_steps, dt):
        self.queue.append((current_time + (delay_steps * dt), data))
    def receive(self, current_time):
        # 1. Initialize latest to None so it is always defined
        latest = None

        # 2. Retrieve data from the queue
        # REVISED ASSUMPTION: self.queue is likely [timestamp, data].
        # The previous error showed that index [1] was an array (causing the crash), 
        # so index [0] must be the timestamp.
        while len(self.queue) > 0:
            # Check timestamp at index 0
            if self.queue[0][0] > current_time:
                break
            
            # Take the packet out of the queue
            # FIX: Since self.queue is a 'deque', pop(0) fails.
            # We must use popleft() to remove the first element.
            packet = self.queue.popleft()
            
            # Extract data from index 1
            latest = packet[1]

        # 3. The Fix: Explicitly check for None
        # Use 'is not None' because 'latest' is a numpy array.
        if latest is not None:
            return latest, current_time

        return None, None  

class LeaderRobot:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.ik_solver = IKSolver(self.model, cfg.JOINT_LIMITS_LOWER, cfg.JOINT_LIMITS_UPPER)
        try:
            self.ee_id = self.model.body('panda_hand').id
            self.ee_type = 'body'
        except KeyError:
            self.ee_id = self.model.site('panda_ee_site').id
            self.ee_type = 'site'
        self.center = cfg.TRAJECTORY_CENTER
        self.scale = cfg.TRAJECTORY_SCALE
        self.freq = cfg.TRAJECTORY_FREQUENCY
        self.q = cfg.INITIAL_JOINT_CONFIG.copy()
        self.q_prev = self.q.copy()
        self.data.qpos[:cfg.N_JOINTS] = self.q
        mujoco.mj_forward(self.model, self.data)
        self.ik_solver.reset_trajectory(self.q)

    def step(self, t, dt):
        if t < cfg.WARM_UP_DURATION: target = self.center
        else:
            phase = (t - cfg.WARM_UP_DURATION) * self.freq * 2 * np.pi
            target = self.center + np.array([self.scale[0]*np.sin(phase), self.scale[1]*np.sin(phase/2), self.scale[2]*np.sin(phase)])
        
        q_tgt, success, _ = self.ik_solver.solve(target, self.q)
        if not success: q_tgt = self.q
        
        qd = (q_tgt - self.q_prev) / dt
        self.q_prev = self.q.copy()
        self.q = q_tgt.copy()
        self.data.qpos[:cfg.N_JOINTS] = self.q
        mujoco.mj_kinematics(self.model, self.data)
        return self.q, qd, self.data.xpos[self.ee_id].copy() if self.ee_type == 'body' else self.data.site_xpos[self.ee_id].copy()

class RemoteRobot:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        try:
            self.ee_id = self.model.body('panda_hand').id
            self.ee_type = 'body'
        except KeyError:
            self.ee_id = self.model.site('panda_ee_site').id
            self.ee_type = 'site'
        self.data.qpos[:cfg.N_JOINTS] = cfg.INITIAL_JOINT_CONFIG
        mujoco.mj_step(self.model, self.data)

    def step(self, target_q, target_qd, tau_rl):
        q, qd = self.data.qpos[:cfg.N_JOINTS], self.data.qvel[:cfg.N_JOINTS]
        acc_des = cfg.DEFAULT_KP_REMOTE * (target_q - q) + cfg.DEFAULT_KD_REMOTE * (target_qd - qd)
        self.data.qpos[:cfg.N_JOINTS], self.data.qvel[:cfg.N_JOINTS], self.data.qacc[:cfg.N_JOINTS] = q, qd, acc_des
        mujoco.mj_inverse(self.model, self.data)
        tau = np.clip(self.data.qfrc_inverse[:cfg.N_JOINTS] + tau_rl, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
        self.data.ctrl[:cfg.N_JOINTS] = tau
        mujoco.mj_step(self.model, self.data)
        return self.data.xpos[self.ee_id].copy() if self.ee_type == 'body' else self.data.site_xpos[self.ee_id].copy()

class SimPublisher(Node):
    def __init__(self):
        super().__init__('combined_sim_node')
        self.pub_lead = self.create_publisher(PointStamped, '/leader/ee_position', 10)
        self.pub_rem = self.create_publisher(PointStamped, '/remote/ee_position', 10)
        self.pub_delay = self.create_publisher(Float32, '/simulation/delay_steps', 10)

    def publish(self, l_pos, r_pos, delay):
        h = Header(stamp=self.get_clock().now().to_msg(), frame_id="world")
        self.pub_lead.publish(PointStamped(header=h, point=self._pt(l_pos)))
        self.pub_rem.publish(PointStamped(header=h, point=self._pt(r_pos)))
        self.pub_delay.publish(Float32(data=float(delay)))

    def _pt(self, p):
        from geometry_msgs.msg import Point
        return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))

# =============================================================================
# MAIN
# =============================================================================
def main():
    rclpy.init()
    leader = LeaderRobot()
    remote = RemoteRobot()
    pub = SimPublisher()
    
    print(f"[Info] Loading Models...")
    predictor = SBSPPredictor(SBSP_MODEL, SBSP_STATS, n_models=1)
    
    actor = Actor(state_dim=cfg.OBS_DIM).to(DEVICE)
    ckpt = torch.load(RL_PATH, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(ckpt['actor_state_dict'] if 'actor_state_dict' in ckpt else ckpt)
    actor.eval()

    remote_hist = deque([np.zeros(14)] * 50, maxlen=50)
    delay_gen = DelaySimulator(CONTROL_FREQ, ExperimentConfig.MEDIUM_DELAY)
    
    # Queue Setup
    feedback_queue = TimeBasedQueue() # Remote -> Agent
    forward_queue = TimeBasedQueue()  # Agent -> Remote

    dt, sim_time, steps = 1.0/CONTROL_FREQ, 0.0, 0
    action = np.zeros(7)
    default_state = np.concatenate([cfg.INITIAL_JOINT_CONFIG, np.zeros(7)])
    
    print("--- HEADLESS SIMULATION STARTED (Correct Delay Flow) ---")
    
    try:
        while rclpy.ok():
            t0 = time.perf_counter()
            
            # 1. PHYSICS: Leader
            gt_q, gt_qd, l_ee = leader.step(sim_time, dt)
            
            # 2. PHYSICS: Remote (Execute command from Forward Queue)
            cmd_pkt, _ = forward_queue.receive(sim_time)
            if cmd_pkt is None:
                cmd_pkt = {'q': cfg.INITIAL_JOINT_CONFIG, 'qd': np.zeros(7), 'tau': np.zeros(7)}
            r_ee = remote.step(cmd_pkt['q'], cmd_pkt['qd'], cmd_pkt['tau'])
            
            # 3. FEEDBACK: Send Remote State to Agent (with Delay)
            obs_delay = delay_gen.get_observation_delay_steps(len(remote_hist))
            rem_state_true = np.concatenate([remote.data.qpos[:7], remote.data.qvel[:7]])
            feedback_queue.send(rem_state_true, sim_time, obs_delay, dt)
            
            # 4. AGENT: Receive Delayed State
            delayed_rem_state, _ = feedback_queue.receive(sim_time)
            if delayed_rem_state is None: delayed_rem_state = default_state
            
            # 5. PREDICTION (SBSP) - FIX: FORCE 1D INPUTS
            # Flatten inputs to avoid (1, N) vs (N,) mismatch in numpy.concatenate
            predictor.push_action(action.flatten())
            pred_state_raw = predictor.predict(delayed_rem_state.flatten(), obs_delay)
            pred_state = pred_state_raw.flatten() # Ensure output is 1D
            
            # 6. RL POLICY
            remote_hist.append(delayed_rem_state)
            hist = np.concatenate(list(remote_hist)[-cfg.REMOTE_HISTORY_LEN:])
            
            obs = np.concatenate([
                delayed_rem_state, 
                hist, 
                pred_state, 
                (pred_state[:7] - delayed_rem_state[:7]), 
                (pred_state[7:] - delayed_rem_state[7:]), 
                [obs_delay / cfg.DELAY_INPUT_NORM_FACTOR]
            ]).astype(np.float32)
            
            with torch.no_grad():
                action = actor.sample(torch.tensor(obs, device=DEVICE).unsqueeze(0), deterministic=True)[0].cpu().numpy().flatten()
            
            # 7. SEND ACTION (Agent -> Remote)
            act_delay = delay_gen.get_action_delay_steps()
            # Command: Target = Leader GT, Torque = RL
            cmd = {'q': gt_q, 'qd': gt_qd, 'tau': action}
            forward_queue.send(cmd, sim_time, act_delay, dt)

            # 8. PUBLISH & SYNC
            pub.publish(l_ee, r_ee, obs_delay)
            
            steps += 1; sim_time += dt
            if steps % 50 == 0: 
                # Print Error between Prediction and TRUE remote state (Hidden from Agent)
                true_err = np.linalg.norm(pred_state[:7] - rem_state_true[:7])
                print(f"\rStep {steps} | Delay: {obs_delay:02d} | PredErr: {true_err:.4f} | LeadZ: {l_ee[2]:.3f}", end="")
            
            while (time.perf_counter() - t0) < dt: pass

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()