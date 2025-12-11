import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import mujoco
from collections import deque
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import os

from E2E_Teleoperation.E2E_RL.local_robot_simulator import LocalRobotSimulator, TrajectoryType
from E2E_Teleoperation.E2E_RL.remote_robot_simulator import RemoteRobotSimulator
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import E2E_Teleoperation.config.robot_config as cfg


class TeleoperationEnvWithDelay(gym.Env):

    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': cfg.DEFAULT_CONTROL_FREQ}
    
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.HIGH_VARIANCE,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_trajectory: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Rendering setup
        self.render_mode = render_mode
        self.viewer = None
        
        # RL/Env parameters
        self.max_episode_steps = cfg.MAX_EPISODE_STEPS
        self.control_freq = cfg.DEFAULT_CONTROL_FREQ
        self.dt = 1.0 / self.control_freq
        self.current_step = 0
        
        # Robot parameters
        self.n_joints = cfg.N_JOINTS
        self.initial_qpos = cfg.INITIAL_JOINT_CONFIG.copy()
        self.max_joint_error = cfg.MAX_JOINT_ERROR_TERMINATION
        self.joint_limits_lower = cfg.JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = cfg.JOINT_LIMITS_UPPER.copy()
        
        # Delay configuration
        self.delay_config = delay_config
        self.delay_simulator = DelaySimulator(control_freq=self.control_freq, config=delay_config, seed=seed)
        
        # Simulators
        self.leader = LocalRobotSimulator(trajectory_type=trajectory_type, randomize_params=randomize_trajectory)
        
        should_render_remote = (self.render_mode == "human")
        self.remote_robot = RemoteRobotSimulator(
            delay_config=delay_config,
            seed=seed,
            render=should_render_remote
        )
        
        # =====================================================================
        # MuJoCo model for Inverse Dynamics Teacher
        # =====================================================================
        self._teacher_model = self.remote_robot.model
        self._teacher_data = mujoco.MjData(self._teacher_model)
        
        # PD gains for computed torque control
        self._teacher_kp = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float64)
        self._teacher_kd = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 10.0], dtype=np.float64)
        
        # =====================================================================
        # [NEW] Separate data for gravity compensation calculation
        # =====================================================================
        self._gravity_data = mujoco.MjData(self._teacher_model)
        # =====================================================================
        
        # Buffers
        self.leader_q_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.remote_q_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        self.remote_qd_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        
        # Pre-computed Trajectory
        self._precomputed_trajectory_q: Optional[np.ndarray] = None
        self._precomputed_trajectory_qd: Optional[np.ndarray] = None
        
        # Helper for storing prediction between calls
        self._current_predicted_state = None 

        # Warmup Logic
        self.warmup_steps = int(cfg.WARM_UP_DURATION * self.control_freq)
        self.steps_remaining_in_warmup = 0
        self.grace_period_steps = self.warmup_steps + int(cfg.NO_DELAY_DURATION * self.control_freq)
        
        ######################################################################
        # Construct RL space
        # Action is RESIDUAL torque on top of gravity compensation
        self.action_space = spaces.Box(
            low=-cfg.MAX_TORQUE_COMPENSATION, high=cfg.MAX_TORQUE_COMPENSATION, 
            shape=(self.n_joints,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.OBS_DIM,), dtype=np.float32)
        ######################################################################
        
        self.last_target_q = None
        self._cached_prediction_error = 0.0
        self._cached_delay_steps = 0
        self._last_gravity_torque = np.zeros(self.n_joints)
        
    #######################################################################
    # Normalization functions
    #######################################################################
    def _normalize_state(self, q, qd):
        """Normalize State (14D) to ~ N(0, 1)"""
        q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
        qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
        return np.concatenate([q_norm, qd_norm])

    def _normalize_input(self, q, qd, delay_scalar):
        """Normalize Input (15D = 14D State + 1D Delay)"""
        state_norm = self._normalize_state(q, qd)
        return np.concatenate([state_norm, [delay_scalar]])
    
    #######################################################################
    # Gravity compensation
    #######################################################################
    def _compute_gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torque for current configuration.
        This provides a baseline that keeps the robot from falling.
        """
        self._gravity_data.qpos[:self.n_joints] = q
        self._gravity_data.qvel[:self.n_joints] = 0.0
        self._gravity_data.qacc[:self.n_joints] = 0.0
        
        mujoco.mj_inverse(self._teacher_model, self._gravity_data)
        
        return self._gravity_data.qfrc_inverse[:self.n_joints].copy()
    
    #######################################################################
    # Inverse dynamics teacher
    #######################################################################
    def _compute_inverse_dynamics_torque(
        self, 
        current_q: np.ndarray, 
        current_qd: np.ndarray,
        target_q: np.ndarray, 
        target_qd: np.ndarray
    ) -> np.ndarray:
        """
        Compute the expert torque using Inverse Dynamics.
        Returns the RESIDUAL torque (full ID torque minus gravity compensation).
        """
        # Compute position and velocity errors
        pos_error = target_q - current_q
        vel_error = target_qd - current_qd
        
        # Compute desired acceleration using PD control law
        qdd_desired = self._teacher_kp * pos_error + self._teacher_kd * vel_error
        
        # Set up MuJoCo data for inverse dynamics
        self._teacher_data.qpos[:self.n_joints] = current_q
        self._teacher_data.qvel[:self.n_joints] = current_qd
        self._teacher_data.qacc[:self.n_joints] = qdd_desired
        
        # Run MuJoCo inverse dynamics
        mujoco.mj_inverse(self._teacher_model, self._teacher_data)
        full_torque = self._teacher_data.qfrc_inverse[:self.n_joints].copy()
        
        # Compute gravity compensation at current config
        gravity_torque = self._compute_gravity_compensation(current_q)
        
        # Return RESIDUAL torque (what RL should output on top of gravity comp)
        residual_torque = full_torque - gravity_torque
        
        return residual_torque
    
    def set_predicted_state(self, predicted_state: np.ndarray):
        """Helper to receive prediction from Agent before step() is called."""
        self._current_predicted_state = predicted_state

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self._current_predicted_state = None
        
        # Reset Signal Processing
        self._cached_prediction_error = 0.0
        self._cached_delay_steps = 0
        
        # Reset Leader & Precompute
        leader_start_q, _ = self.leader.reset(seed=seed)
        self._precompute_trajectory(leader_start_q)
        
        # Reset Remote
        self.remote_robot.reset(initial_qpos=self.initial_qpos)
        
        # Clear Buffers
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        self.remote_q_history.clear()
        self.remote_qd_history.clear()
        
        # Prefill Buffers (Warmup State)
        for _ in range(cfg.RNN_SEQUENCE_LENGTH + 20):
            self.leader_q_history.append(leader_start_q.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))
        
        start_q = leader_start_q.copy()
        for _ in range(cfg.REMOTE_HISTORY_LEN):
            self.remote_q_history.append(start_q.copy())
            self.remote_qd_history.append(np.zeros(self.n_joints))
            
        self.steps_remaining_in_warmup = self.warmup_steps
        self.last_target_q = start_q.copy()
        
        # Initialize gravity compensation
        self._last_gravity_torque = self._compute_gravity_compensation(self.initial_qpos)
        
        return self._get_observation(), {}

    def _precompute_trajectory(self, start_q):
        """Simulate the leader forward to get ground truth for the whole episode."""
        rollout_steps = self.max_episode_steps + cfg.ESTIMATOR_PREDICTION_HORIZON
        
        backup_state = (self.leader._q_current.copy(), self.leader._q_previous.copy(), 
                       self.leader._trajectory_time, self.leader._tick)
        
        temp_q = [start_q.copy()]
        temp_qd = [np.zeros(self.n_joints)]
        
        for _ in range(rollout_steps):
            q, qd, _, _, _, _ = self.leader.step()
            temp_q.append(q.copy())
            temp_qd.append(qd.copy())
            
        self._precomputed_trajectory_q = np.array(temp_q)
        self._precomputed_trajectory_qd = np.array(temp_qd)
        
        # Restore leader state
        self.leader._q_current = backup_state[0]
        self.leader._q_previous = backup_state[1]
        self.leader._trajectory_time = backup_state[2]
        self.leader._tick = backup_state[3]

    def step(self, action: np.ndarray):
        self.current_step += 1
        
        # 1. Step Leader (Ground Truth Generator)
        new_leader_q, new_leader_qd, _, _, _, _ = self.leader.step()
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())

        # 2. Get current remote state for gravity compensation
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        
        # 3. Compute gravity compensation (baseline torque)
        gravity_torque = self._compute_gravity_compensation(remote_q)
        self._last_gravity_torque = gravity_torque
        
        # 4. Determine target for remote
        true_target_full = self.get_true_current_target()
        target_q_gt = true_target_full[:self.n_joints]
        target_qd_gt = true_target_full[self.n_joints:]

        # 5. Compute total torque: gravity compensation + RL residual
        if self.steps_remaining_in_warmup > 0:
            self.steps_remaining_in_warmup -= 1
            tau_total = gravity_torque  # Only gravity comp during warmup
        else:
            tau_total = gravity_torque + action  # Gravity comp + RL residual

        # Use the stored prediction if available
        pred_q_val = None
        if self._current_predicted_state is not None:
             pred_q_val = self._current_predicted_state[:self.n_joints]

        # 6. Step Remote Robot with TOTAL torque
        self.remote_robot.step(
            target_q=target_q_gt,   
            target_qd=target_qd_gt, 
            torque_input=tau_total,  # Send total torque
            true_local_q=target_q_gt,
            predicted_q=pred_q_val
        )
        
        # 7. Reward & Info
        reward, r_tracking = self._calculate_reward(action)
        remote_q_new, _ = self.remote_robot.get_joint_state()
        
        # Stats
        joint_error = np.linalg.norm(target_q_gt - remote_q_new)
        
        # Delay Stats
        hist_len = len(self.leader_q_history)
        self._cached_delay_steps = 0 if self.current_step < self.grace_period_steps else \
                                   self.delay_simulator.get_state_delay_steps(hist_len)

        terminated, term_penalty = self._check_termination(joint_error, remote_q_new)
        reward += term_penalty
        
        truncated = self.current_step >= self.max_episode_steps
        if self.render_mode == "human": self.render()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def get_delayed_target_buffer(self, buffer_length: int) -> np.ndarray:
        history_len = len(self.leader_q_history)
        delay_steps = self.delay_simulator.get_state_delay_steps(history_len)
        normalized_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        most_recent_idx = -1 - delay_steps
        
        buffer_seq = []
        for i in range(-buffer_length + 1, 1): 
            idx = np.clip(most_recent_idx + i, -history_len, -1)
            step_vector = self._normalize_input(
                self.leader_q_history[idx], 
                self.leader_qd_history[idx], 
                normalized_delay
            )
            buffer_seq.append(step_vector)
        return np.array(buffer_seq).flatten().astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        self.remote_q_history.append(remote_q.copy())
        self.remote_qd_history.append(remote_qd.copy())
        
        # 1. Normalize Current State
        remote_state_normalized = self._normalize_state(remote_q, remote_qd)
        
        # 2. Flatten Robot History (Recent Past)
        # We grab the last N steps of the robot's own state
        # using the same length as the target buffer (RNN_SEQUENCE_LENGTH)
        robot_history_seq = []
        hist_len = len(self.remote_q_history)
        
        # Iterate backwards to get the recent history
        for i in range(-cfg.RNN_SEQUENCE_LENGTH, 0):
            # Safe indexing with clamping
            idx = max(-hist_len, i)
            # We don't need delay scalar for the robot's own state, just q and qd
            q_norm = (self.remote_q_history[idx] - cfg.Q_MEAN) / cfg.Q_STD
            qd_norm = (self.remote_qd_history[idx] - cfg.QD_MEAN) / cfg.QD_STD
            robot_history_seq.extend(q_norm)
            robot_history_seq.extend(qd_norm)

        robot_history_flat = np.array(robot_history_seq, dtype=np.float32)
        
        # 3. Target History (Existing logic)
        target_history_flat = self.get_delayed_target_buffer(cfg.RNN_SEQUENCE_LENGTH)
        
        # Combine: [Current State] + [Robot History] + [Target History]
        obs = np.concatenate([
            remote_state_normalized, 
            robot_history_flat, 
            target_history_flat
        ]).astype(np.float32)
        
        return obs

    def get_true_current_target(self) -> np.ndarray:
        if not self.leader_q_history:
            return np.concatenate([self.initial_qpos, np.zeros(self.n_joints)])
        return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])

    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, float]:
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()
        target_q = true_target[:self.n_joints]
        target_qd = true_target[self.n_joints:]
        
        pos_error = np.linalg.norm(target_q - remote_q)
        robot_speed = np.linalg.norm(remote_qd)
        
        # Position tracking
        r_pos = 1.0 * np.exp(-5.0 * pos_error)
        
        # LAZY PENALTY: If there's error but robot isn't moving, penalize heavily
        if pos_error > 0.02 and robot_speed < 0.01:
            r_lazy = -2.0  # Strong penalty
        else:
            r_lazy = 0.0
        
        total_reward = r_pos + r_lazy
        
        return float(total_reward), float(r_pos)

    def _check_termination(self, joint_error: float, remote_q: np.ndarray) -> Tuple[bool, float]:
        if not np.all(np.isfinite(remote_q)): return True, -100.0
        at_limits = (np.any(remote_q <= self.joint_limits_lower + cfg.JOINT_LIMIT_MARGIN) or 
                    np.any(remote_q >= self.joint_limits_upper - cfg.JOINT_LIMIT_MARGIN))
        high_error = joint_error > self.max_joint_error
        terminated = at_limits or high_error
        penalty = -10.0 if terminated else 0.0
        return terminated, penalty

    def _get_info(self, phase="unknown"):
        is_warmup = self.current_step < self.grace_period_steps
        true_state = self.get_true_current_target()
        return {
            "prediction_error": 0.0,
            "current_delay_steps": self._cached_delay_steps,
            "is_in_warmup": is_warmup,
            "true_state": true_state,
            "gravity_torque": self._last_gravity_torque.copy()
        }

    def render(self):
        pass
        
    def close(self):
        if self.viewer is not None: plt.close(self.viewer)
