"""
Pure Python Teleoperation Simulation - PD CONTROL BASELINE (No AI)

DELAY MODEL (Matching Training Environment):
============================================

In the TRAINING environment:
1. Observation Delay: Leader state is delayed before LSTM sees it
2. LSTM predicts current state from delayed observations  
3. PD controller uses LSTM prediction IMMEDIATELY (no additional delay)
4. RL torque compensation is queued with action delay

In this BASELINE (No LSTM):
1. Observation Delay: Leader state is delayed before remote sees it
2. NO prediction - remote uses delayed state directly as target
3. PD controller computes torque from delayed target
4. Action Delay: PD torque is queued before execution

This represents what happens WITHOUT the LSTM prediction capability.
The RL agent's job is to compensate for the prediction errors that remain
after LSTM prediction, so the baseline shows raw delay impact.
"""

import numpy as np
import time
import mujoco
import mujoco.viewer
from collections import deque
from dataclasses import dataclass, field
import heapq
from typing import Optional, Tuple, List

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
EE_POSE_PUBLISH_FREQ = 50  # Hz
METRICS_PUBLISH_FREQ = 50  # Hz


# =============================================================================
# 1. LEADER ROBOT (Local/Master) - Unchanged
# =============================================================================
@dataclass(frozen=True)
class TrajectoryParams:
    center: np.ndarray = field(default_factory=lambda: cfg.TRAJECTORY_CENTER.copy())
    scale: np.ndarray = field(default_factory=lambda: cfg.TRAJECTORY_SCALE.copy())
    frequency: float = cfg.TRAJECTORY_FREQUENCY
    initial_phase: float = 0.0


class Figure8Trajectory:
    def __init__(self, params: TrajectoryParams):
        self._params = params

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
        
        # Handle different naming conventions
        try:
            self.ee_site_id = self.model.site('panda_ee_site').id
        except:
            self.ee_site_id = self.model.site(cfg.EE_BODY_NAME.replace("body", "site")).id

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

    def step(self, t: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        if t < cfg.WARM_UP_DURATION:
            alpha = t / cfg.WARM_UP_DURATION
            target_pos = (1 - alpha) * self.start_ee_pos + alpha * self.traj_start_ee_pos
        else:
            target_pos = self.generator.compute_position(t - cfg.WARM_UP_DURATION)

        q_target, success, _ = self.ik_solver.solve(target_pos, self.q)
        if not success or q_target is None:
            q_target = self.q.copy()

        self.qd = (q_target - self.q_prev) / dt
        self.q_prev = self.q.copy()
        self.q = q_target.copy()

        self.data.qpos[:cfg.N_JOINTS] = self.q
        mujoco.mj_kinematics(self.model, self.data)

        return self.q.copy(), self.qd.copy()

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()


# =============================================================================
# 2. REMOTE ROBOT - PD Baseline with Proper Dual-Channel Delay
# =============================================================================
class RemoteRobotBaseline:
    """
    Remote robot baseline WITHOUT LSTM prediction.
    
    Delay Channels:
    1. Observation Buffer: Stores leader states, accessed with obs_delay
    2. Action Queue: PD torque commands delayed by act_delay before execution
    
    This models what the system would do without any state prediction.
    """

    def __init__(
        self,
        delay_config: ExperimentConfig,
        seed: Optional[int] = None,
        render: bool = True
    ):
        # MuJoCo Setup
        self.model = mujoco.MjModel.from_xml_path(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.data_ctrl = mujoco.MjData(self.model)  # For inverse dynamics
        
        try:
            self.ee_site_id = self.model.site('panda_ee_site').id
        except:
            self.ee_site_id = self.model.site(cfg.EE_BODY_NAME.replace("body", "site")).id

        # Control parameters
        self.control_freq = cfg.DEFAULT_CONTROL_FREQ
        self.dt = 1.0 / self.control_freq
        sim_freq = int(1.0 / self.model.opt.timestep)
        self.n_substeps = sim_freq // self.control_freq
        self.n_joints = cfg.N_JOINTS

        # PD Gains - Match training (remote_robot_simulator.py line 51-52)
        self.kp = cfg.DEFAULT_KP_REMOTE * 0.5
        self.kd = cfg.DEFAULT_KD_REMOTE * 0.5

        # Delay Simulator
        self.delay_sim = DelaySimulator(self.control_freq, config=delay_config, seed=seed)

        # Grace Period - Match training (remote_robot_simulator.py line 63-64)
        total_grace_time = cfg.WARM_UP_DURATION + cfg.NO_DELAY_DURATION
        self.no_delay_steps = int(total_grace_time * self.control_freq)

        # =============================================
        # CHANNEL 1: Observation Delay (Ring Buffer)
        # =============================================
        # Match training_env.py: leader_q_history buffer
        max_delay_steps = int(0.5 * self.control_freq)  # 500ms max
        self.obs_buffer_size = max_delay_steps + 50  # Safety margin
        self.obs_buffer_q = np.zeros((self.obs_buffer_size, self.n_joints))
        self.obs_buffer_qd = np.zeros((self.obs_buffer_size, self.n_joints))
        self.write_idx = 0  # Current write position

        # =============================================
        # CHANNEL 2: Action Delay (Priority Queue)
        # =============================================
        # Match remote_robot_simulator.py: action_queue with heapq
        self.action_queue: List[Tuple[int, np.ndarray]] = []
        heapq.heapify(self.action_queue)
        self.last_executed_torque = np.zeros(self.n_joints)

        # State tracking
        self.internal_tick = 0
        self.current_obs_delay = 0
        self.current_act_delay = 0

        # Rendering
        self._render_enabled = render
        self._viewer = None

        # Initialize
        self._reset_robot(cfg.INITIAL_JOINT_CONFIG)

    def _reset_robot(self, q_init: np.ndarray):
        """Reset robot state and buffers."""
        # Reset physics state
        self.data.qpos[:self.n_joints] = q_init
        self.data.qvel[:self.n_joints] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Compute gravity compensation
        self.data_ctrl.qpos[:self.n_joints] = q_init
        self.data_ctrl.qvel[:self.n_joints] = 0.0
        self.data_ctrl.qacc[:self.n_joints] = 0.0
        mujoco.mj_inverse(self.model, self.data_ctrl)
        gravity_torque = self.data_ctrl.qfrc_inverse[:self.n_joints].copy()

        # Settling loop (match remote_robot_simulator.py line 137-140)
        for _ in range(100):
            self.data.ctrl[:self.n_joints] = gravity_torque
            self.data.qvel[:self.n_joints] = 0.0
            mujoco.mj_step(self.model, self.data)

        # Initialize observation buffer with initial state
        for i in range(self.obs_buffer_size):
            self.obs_buffer_q[i] = q_init.copy()
            self.obs_buffer_qd[i] = np.zeros(self.n_joints)

        # Reset counters
        self.internal_tick = 0
        self.write_idx = 0
        self.action_queue = []
        heapq.heapify(self.action_queue)
        self.last_executed_torque = np.zeros(self.n_joints)

    def receive_leader_state(self, q: np.ndarray, qd: np.ndarray):
        """
        Store leader state in observation buffer.
        Called every step with CURRENT leader state.
        """
        idx = self.write_idx % self.obs_buffer_size
        self.obs_buffer_q[idx] = q.copy()
        self.obs_buffer_qd[idx] = qd.copy()
        self.write_idx += 1

    def _get_delayed_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get delayed leader state from observation buffer.
        
        Match training_env.py line 294:
        delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
        """
        history_len = self.write_idx  # How many samples we have

        if self.internal_tick < self.no_delay_steps:
            # Grace period: no delay
            self.current_obs_delay = 0
        else:
            # Get stochastic observation delay
            self.current_obs_delay = self.delay_sim.get_observation_delay_steps(history_len)

        # Calculate which buffer index to read from
        # write_idx points to NEXT write position, so latest data is at write_idx - 1
        latest_idx = self.write_idx - 1
        delayed_idx = max(0, latest_idx - self.current_obs_delay)
        buffer_idx = delayed_idx % self.obs_buffer_size

        return self.obs_buffer_q[buffer_idx].copy(), self.obs_buffer_qd[buffer_idx].copy()

    def _normalize_angle(self, angle: np.ndarray) -> np.ndarray:
        """Normalize angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _compute_inverse_dynamics(
        self, q: np.ndarray, qd: np.ndarray, qacc: np.ndarray
    ) -> np.ndarray:
        """Compute torque via inverse dynamics."""
        self.data_ctrl.qpos[:self.n_joints] = q
        self.data_ctrl.qvel[:self.n_joints] = qd
        self.data_ctrl.qacc[:self.n_joints] = qacc
        mujoco.mj_inverse(self.model, self.data_ctrl)
        return self.data_ctrl.qfrc_inverse[:self.n_joints].copy()

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute one control step with dual-channel delay.
        
        Returns:
            Tuple of (current_q, current_qd)
        """
        self.internal_tick += 1

        # =============================================
        # 1. Get delayed observation (Channel 1)
        # =============================================
        target_q, target_qd = self._get_delayed_observation()

        # =============================================
        # 2. Compute PD control torque
        # =============================================
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()

        # PD control (match remote_robot_simulator.py line 183-185)
        q_error = self._normalize_angle(target_q - q_current)
        qd_error = target_qd - qd_current
        acc_desired = self.kp * q_error + self.kd * qd_error

        # Inverse dynamics for feedforward
        tau_pd = self._compute_inverse_dynamics(q_current, qd_current, acc_desired)
        tau_pd[-1] = 0.0  # Disable last joint (match training)

        # =============================================
        # 3. Queue torque with action delay (Channel 2)
        # =============================================
        if self.internal_tick < self.no_delay_steps:
            # Grace period: instant execution
            self.current_act_delay = 0
            self.last_executed_torque = tau_pd.copy()
        else:
            # Get stochastic action delay
            # Match remote_robot_simulator.py line 172
            self.current_act_delay = int(self.delay_sim.get_action_delay_steps())
            
            arrival_tick = self.internal_tick + self.current_act_delay
            heapq.heappush(self.action_queue, (arrival_tick, tau_pd.copy()))

            # Process arrived commands (match line 177-178)
            while self.action_queue and self.action_queue[0][0] <= self.internal_tick:
                _, arrived_torque = heapq.heappop(self.action_queue)
                self.last_executed_torque = arrived_torque

        # =============================================
        # 4. Apply torque and step physics
        # =============================================
        tau_clipped = np.clip(
            self.last_executed_torque,
            -cfg.TORQUE_LIMITS,
            cfg.TORQUE_LIMITS
        )
        self.data.ctrl[:self.n_joints] = tau_clipped

        # Sub-stepping (match remote_robot_simulator.py line 195-196)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        return (
            self.data.qpos[:self.n_joints].copy(),
            self.data.qvel[:self.n_joints].copy()
        )

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_delays(self) -> Tuple[int, int]:
        """Return current delay values."""
        return self.current_obs_delay, self.current_act_delay

    def init_viewer(self):
        """Initialize viewer if rendering enabled."""
        if self._render_enabled and self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.azimuth = 135
            self._viewer.cam.elevation = -20
            self._viewer.cam.distance = 2.0
            self._viewer.cam.lookat[:] = [0.4, 0.0, 0.4]

    def sync_viewer(self) -> bool:
        """Sync viewer, return False if closed."""
        if self._viewer is not None:
            if not self._viewer.is_running():
                return False
            self._viewer.sync()
        return True

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# =============================================================================
# 3. ROS 2 PUBLISHER
# =============================================================================
class SimPublisher(Node):
    def __init__(self, control_freq: int):
        super().__init__('sim_publisher_baseline')

        self.pub_leader_q = self.create_publisher(JointState, 'leader/joint_states', 100)
        self.pub_remote_q = self.create_publisher(JointState, 'remote/joint_states', 100)
        self.pub_leader_ee = self.create_publisher(PointStamped, 'leader/ee_pose', 50)
        self.pub_remote_ee = self.create_publisher(PointStamped, 'remote/ee_pose', 50)
        self.pub_obs_delay = self.create_publisher(Float32, 'agent/obs_delay_steps', 50)
        self.pub_act_delay = self.create_publisher(Float32, 'agent/act_delay_steps', 50)
        self.pub_tracking_error = self.create_publisher(Float32, 'metrics/tracking_error', 50)
        self.pub_ee_error = self.create_publisher(Float32, 'metrics/ee_error', 50)

        self.control_freq = control_freq
        self.ee_skip = max(1, control_freq // EE_POSE_PUBLISH_FREQ)
        self.metrics_skip = max(1, control_freq // METRICS_PUBLISH_FREQ)
        self.step_count = 0

    def publish_all(
        self,
        gt_q: np.ndarray,
        rem_q: np.ndarray,
        leader_ee: np.ndarray,
        remote_ee: np.ndarray,
        obs_delay: int,
        act_delay: int,
        tracking_error: float,
        ee_error: float
    ):
        now = self.get_clock().now().to_msg()
        self.step_count += 1

        # Joint states (always publish)
        def create_js(q):
            msg = JointState()
            msg.header.stamp = now
            msg.name = [f'panda_joint{i+1}' for i in range(7)]
            msg.position = q.tolist()
            return msg

        self.pub_leader_q.publish(create_js(gt_q))
        self.pub_remote_q.publish(create_js(rem_q))

        # EE poses (throttled)
        if self.step_count % self.ee_skip == 0:
            def create_ps(pos):
                msg = PointStamped()
                msg.header.stamp = now
                msg.header.frame_id = "world"
                msg.point.x = float(pos[0])
                msg.point.y = float(pos[1])
                msg.point.z = float(pos[2])
                return msg

            self.pub_leader_ee.publish(create_ps(leader_ee))
            self.pub_remote_ee.publish(create_ps(remote_ee))

        # Metrics (throttled)
        if self.step_count % self.metrics_skip == 0:
            obs_msg = Float32()
            obs_msg.data = float(obs_delay)
            self.pub_obs_delay.publish(obs_msg)

            act_msg = Float32()
            act_msg.data = float(act_delay)
            self.pub_act_delay.publish(act_msg)

            err_msg = Float32()
            err_msg.data = float(tracking_error)
            self.pub_tracking_error.publish(err_msg)

            ee_msg = Float32()
            ee_msg.data = float(ee_error)
            self.pub_ee_error.publish(ee_msg)


# =============================================================================
# 4. MAIN LOOP
# =============================================================================
def main(args=None):
    rclpy.init(args=args)

    control_freq = cfg.DEFAULT_CONTROL_FREQ
    dt = 1.0 / control_freq

    publisher = SimPublisher(control_freq)
    leader = LeaderRobot()

    # Use same delay config as training
    delay_config = ExperimentConfig.MEDIUM_DELAY
    remote = RemoteRobotBaseline(delay_config=delay_config, seed=42, render=True)
    remote.init_viewer()

    print("=" * 70)
    print("TELEOPERATION BASELINE - PD Control WITHOUT LSTM Prediction")
    print("=" * 70)
    print(f"  Control Frequency: {control_freq} Hz")
    print(f"  Delay Config: {delay_config.name}")
    print(f"  Grace Period: {cfg.WARM_UP_DURATION + cfg.NO_DELAY_DURATION:.1f}s")
    print()
    print("  Delay Model (Matching Training Environment):")
    print("    Channel 1 - Observation: Leader state delayed by d_obs steps")
    print("    Channel 2 - Action: PD torque delayed by d_act steps")
    print()
    print("  This baseline shows tracking error WITHOUT LSTM prediction.")
    print("  Compare against RL agent to measure improvement from LSTM + RL.")
    print("=" * 70)

    sim_time = 0.0
    steps = 0

    # Metrics accumulation
    tracking_errors = []
    ee_errors = []

    try:
        while rclpy.ok():
            step_start = time.perf_counter()

            # 1. Leader generates current state
            gt_q, gt_qd = leader.step(sim_time, dt)
            leader_ee = leader.get_ee_pos()

            # 2. Leader state enters observation buffer
            remote.receive_leader_state(gt_q, gt_qd)

            # 3. Remote executes PD control with delays
            rem_q, rem_qd = remote.step()
            remote_ee = remote.get_ee_pos()

            # 4. Compute metrics (error vs CURRENT leader, not delayed)
            q_error_vec = (rem_q - gt_q + np.pi) % (2 * np.pi) - np.pi
            tracking_error = np.linalg.norm(q_error_vec)
            ee_error = np.linalg.norm(remote_ee - leader_ee)

            obs_delay, act_delay = remote.get_delays()

            # 5. Publish
            publisher.publish_all(
                gt_q, rem_q, leader_ee, remote_ee,
                obs_delay, act_delay, tracking_error, ee_error
            )

            # 6. Viewer sync
            if not remote.sync_viewer():
                print("Viewer closed.")
                break

            steps += 1
            sim_time += dt

            # Real-time sync
            elapsed = time.perf_counter() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            # Logging
            if steps % 100 == 0:
                tracking_errors.append(tracking_error)
                ee_errors.append(ee_error)

                total_delay_ms = (obs_delay + act_delay) * dt * 1000
                queue_len = len(remote.action_queue)

                print(
                    f"[Step {steps:5d}] "
                    f"Track: {tracking_error:.4f} rad | "
                    f"EE: {ee_error:.4f} m | "
                    f"Delay: obs={obs_delay}, act={act_delay} ({total_delay_ms:.0f}ms) | "
                    f"Queue: {queue_len}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Statistics
        if tracking_errors:
            print()
            print("=" * 70)
            print("BASELINE STATISTICS")
            print("=" * 70)
            print(f"  Total Steps: {steps}")
            print(f"  Mean Tracking Error: {np.mean(tracking_errors):.4f} rad")
            print(f"  Std Tracking Error:  {np.std(tracking_errors):.4f} rad")
            print(f"  Max Tracking Error:  {np.max(tracking_errors):.4f} rad")
            print(f"  Mean EE Error: {np.mean(ee_errors):.4f} m")
            print(f"  Max EE Error:  {np.max(ee_errors):.4f} m")
            print("=" * 70)

        remote.close()
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()