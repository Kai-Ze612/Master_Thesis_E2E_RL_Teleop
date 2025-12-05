"""
Pure Python Teleoperation Simulation with ROS 2 Publishing.
- NO AI AGENT - Basic PD control with delayed states
- Remote robot directly tracks delayed leader states
- Remote has NO signal until first delayed packet arrives
- Publishes State, EE Pose, and DELAY METRICS.
- EE Pose and Delay Metrics at 50 Hz (reduced from control freq)
"""

import numpy as np
import time
import mujoco
import mujoco.viewer
from collections import deque
from dataclasses import dataclass, field

# ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

# Project Imports
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.inverse_kinematics import IKSolver
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
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
# 2. REMOTE ROBOT (Basic PD Control - No AI)
# =============================================================================
class RemoteRobot:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = self.model.site(cfg.EE_BODY_NAME.replace("body", "site") if "body" in cfg.EE_BODY_NAME else "panda_ee_site").id
        
        self.data.qpos[:cfg.N_JOINTS] = cfg.INITIAL_JOINT_CONFIG
        mujoco.mj_step(self.model, self.data)

    def step(self, target_q, target_qd):
        """
        Basic PD control - NO RL torque compensation.
        Directly tracks the delayed leader states.
        """
        q = self.data.qpos[:cfg.N_JOINTS].copy()
        qd = self.data.qvel[:cfg.N_JOINTS].copy()
       
        # PD control
        q_err = target_q - q
        qd_err = target_qd - qd
        acc_des = cfg.DEFAULT_KP_REMOTE * q_err + cfg.DEFAULT_KD_REMOTE * qd_err
        
        # Inverse dynamics
        self.data.qpos[:cfg.N_JOINTS] = q
        self.data.qvel[:cfg.N_JOINTS] = qd
        self.data.qacc[:cfg.N_JOINTS] = acc_des
        mujoco.mj_inverse(self.model, self.data)
        tau_id = self.data.qfrc_inverse[:cfg.N_JOINTS].copy()
        
        # Apply torque (NO RL compensation)
        tau = np.clip(tau_id, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
        tau[-1] = 0.0
        
        self.data.ctrl[:cfg.N_JOINTS] = tau
        mujoco.mj_step(self.model, self.data)
        return q, qd
    
    def step_hold(self):
        """
        Hold current position when no signal received.
        Just run physics with zero control (or hold position).
        """
        q = self.data.qpos[:cfg.N_JOINTS].copy()
        qd = self.data.qvel[:cfg.N_JOINTS].copy()
        
        # Hold current position with PD control
        q_err = cfg.INITIAL_JOINT_CONFIG - q  # Hold at initial config
        qd_err = -qd  # Damp velocity
        acc_des = cfg.DEFAULT_KP_REMOTE * q_err + cfg.DEFAULT_KD_REMOTE * qd_err
        
        self.data.qpos[:cfg.N_JOINTS] = q
        self.data.qvel[:cfg.N_JOINTS] = qd
        self.data.qacc[:cfg.N_JOINTS] = acc_des
        mujoco.mj_inverse(self.model, self.data)
        tau_id = self.data.qfrc_inverse[:cfg.N_JOINTS].copy()
        
        tau = np.clip(tau_id, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
        tau[-1] = 0.0
        
        self.data.ctrl[:cfg.N_JOINTS] = tau
        mujoco.mj_step(self.model, self.data)
        return q, qd

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

# =============================================================================
# 3. ROS 2 PUBLISHER (WITH RATE CONTROL)
# =============================================================================
class SimPublisher(Node):
    def __init__(self, control_freq: int):
        super().__init__('sim_publisher')
        
        # Full-rate publishers (joint states)
        self.pub_leader_q = self.create_publisher(JointState, 'leader/joint_states', 100)
        self.pub_remote_q = self.create_publisher(JointState, 'remote/joint_states', 100)
        
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

    def publish_all(self, gt_q, rem_q, leader_ee, remote_ee, obs_delay, act_delay):
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
# 4. MAIN LOOP (No AI - Basic PD Control with Proper Delay)
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    
    control_freq = cfg.DEFAULT_CONTROL_FREQ
    publisher = SimPublisher(control_freq)
    
    leader = LeaderRobot()
    remote = RemoteRobot()
    
    dt = 1.0 / control_freq
    delay_sim = DelaySimulator(control_freq, ExperimentConfig.FULL_RANGE_COVER, seed=42)
    
    # Packet queue for observation delay (leader -> remote)
    leader_packet_queue = deque()
    
    # Command queue for action delay (remote command -> execution)
    action_cmd_queue = deque()

    print(f"=" * 60)
    print(f"STARTING SIMULATION (NO AI - BASIC PD CONTROL)")
    print(f"=" * 60)
    print(f"  Control Loop: {control_freq} Hz")
    print(f"  EE Pose Publish: {EE_POSE_PUBLISH_FREQ} Hz")
    print(f"  Delay Metrics Publish: {METRICS_PUBLISH_FREQ} Hz")
    print(f"  Mode: Direct delayed state tracking (no prediction)")
    print(f"=" * 60)
    
    sim_time = 0.0
    steps = 0
    
    # Track if we have received any signal yet
    has_received_signal = False
    last_received_q = None
    last_received_qd = None
    
    with mujoco.viewer.launch_passive(remote.model, remote.data) as viewer:
        while viewer.is_running() and rclpy.ok():
            step_start = time.perf_counter()
            
            # 1. Leader step - generates new state
            gt_q, gt_qd = leader.step(sim_time, dt)
            leader_ee = leader.get_ee_pos()
            
            # 2. Add leader state to packet queue (will be delayed)
            leader_packet_queue.append({
                'q': gt_q.copy(), 
                'qd': gt_qd.copy(), 
                't': sim_time
            })
            
            # 3. Get current observation delay
            obs_delay_steps = delay_sim.get_observation_delay_steps(len(leader_packet_queue))
            obs_delay_time = obs_delay_steps * dt
            
            # 4. Check if any packets have "arrived" (past the delay threshold)
            received_packets = []
            while leader_packet_queue:
                pkt = leader_packet_queue[0]
                packet_age = sim_time - pkt['t']
                if packet_age >= obs_delay_time:
                    received_packets.append(leader_packet_queue.popleft())
                else:
                    break
            
            # 5. Update last received state if we got new packets
            if received_packets:
                has_received_signal = True
                # Use the most recent received packet
                last_received_q = received_packets[-1]['q']
                last_received_qd = received_packets[-1]['qd']
            
            # 6. Get action delay
            act_delay_steps = delay_sim.get_action_delay_steps()
            
            # 7. Remote robot control
            if has_received_signal:
                # We have signal - queue the command for action delay
                action_cmd_queue.append({
                    'q': last_received_q.copy(),
                    'qd': last_received_qd.copy()
                })
                
                # Apply command with action delay
                if act_delay_steps >= len(action_cmd_queue):
                    # Action delay longer than queue - hold position
                    rem_q, rem_qd = remote.step_hold()
                else:
                    # Get delayed command
                    cmd = action_cmd_queue[-1 - act_delay_steps]
                    rem_q, rem_qd = remote.step(cmd['q'], cmd['qd'])
            else:
                # NO SIGNAL YET - remote holds initial position
                rem_q, rem_qd = remote.step_hold()
            
            remote_ee = remote.get_ee_pos()
            
            # 8. Publish ROS 2
            publisher.publish_all(gt_q, rem_q, leader_ee, remote_ee, obs_delay_steps, act_delay_steps)
            
            # 9. Sync & Logging
            steps += 1
            sim_time += dt
            viewer.sync()
            
            elapsed = time.perf_counter() - step_start
            if elapsed < dt: time.sleep(dt - elapsed)
                
            if steps % 20 == 0:
                track_err = np.linalg.norm(rem_q - gt_q)
                total_delay_ms = (obs_delay_steps + act_delay_steps) * dt * 1000
                signal_status = "TRACKING" if has_received_signal else "WAITING (no signal)"
                print(f"[Step {steps}] {signal_status} | Joint Err: {track_err:.4f} | "
                      f"Obs: {obs_delay_steps} | Act: {act_delay_steps} | "
                      f"Total: {total_delay_ms:.0f}ms")

    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()