import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from collections import deque
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import os

from local_robot_simulator import LocalRobotSimulator, TrajectoryType
from remote_robot_simulator import RemoteRobotSimulator
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
        
        # Buffers
        self.leader_q_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.remote_q_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        self.remote_qd_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        
        # Pre-computed Trajectory (for validation/ground truth)
        self._precomputed_trajectory_q: Optional[np.ndarray] = None
        self._precomputed_trajectory_qd: Optional[np.ndarray] = None
        
        # Warmup Logic
        self.warmup_steps = int(cfg.WARM_UP_DURATION * self.control_freq)
        self.steps_remaining_in_warmup = 0
        self.grace_period_steps = self.warmup_steps + int(cfg.NO_DELAY_DURATION * self.control_freq)
        
        # [MODIFICATION] Removed LSTM loading logic entirely
        # The environment is now "dumb" - it just passes raw history to the agent.

        ######################################################################
        # Construct RL space
        self.action_space = spaces.Box(
            low=-cfg.TORQUE_LIMITS, high=cfg.TORQUE_LIMITS, 
            shape=(self.n_joints,), dtype=np.float32
        )
        
        # [MODIFICATION] Observation is now [RemoteState(14) + FlattenedHistory]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.OBS_DIM,), dtype=np.float32)
        
        ######################################################################
        self.last_target_q = None
        self._cached_prediction_error = 0.0
        self._cached_delay_steps = 0
        
    #######################################################################
    # Normalized and denormalized function 
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        
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
        
        return self._get_observation(), {}

    def _precompute_trajectory(self, start_q):
        """Simulate the leader forward to get ground truth for the whole episode."""
        rollout_steps = self.max_episode_steps + cfg.ESTIMATOR_PREDICTION_HORIZON
        
        # Save current leader state
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

        # [MODIFICATION] No LSTM Prediction Step here.
        # The agent receives the raw history and does the prediction internally.
        
        # 2. Determine Safe Target for Remote
        # For End-to-End Joint Training, the "Target" we pass to the simulator 
        # is just the TRUE CURRENT LEADER STATE (Ground Truth).
        # The simulator uses this only for logging error metrics.
        true_target_full = self.get_true_current_target()
        target_q_gt = true_target_full[:self.n_joints]
        target_qd_gt = true_target_full[self.n_joints:]

        if self.steps_remaining_in_warmup > 0:
            self.steps_remaining_in_warmup -= 1
            tau_input = np.zeros(self.n_joints)
        else:
            # RL Action is Direct Torque
            tau_input = action

        # 3. Step Remote Robot
        self.remote_robot.step(
            target_q=target_q_gt,   # Passed for error calculation/logging
            target_qd=target_qd_gt, # Passed for error calculation/logging
            torque_input=tau_input, 
            true_local_q=target_q_gt,
        )
        
        # 4. Reward & Info
        reward, r_tracking = self._calculate_reward(action)
        remote_q, _ = self.remote_robot.get_joint_state()
        
        # Stats
        joint_error = np.linalg.norm(target_q_gt - remote_q)
        
        # Delay Stats
        hist_len = len(self.leader_q_history)
        self._cached_delay_steps = 0 if self.current_step < self.grace_period_steps else \
                                   self.delay_simulator.get_state_delay_steps(hist_len)

        # [MODIFICATION] Prediction error is now calculated by the Agent (via Aux Loss), 
        # but for termination check, we rely purely on Joint Error (Real Physics).
        terminated, term_penalty = self._check_termination(joint_error, remote_q)
        reward += term_penalty
        
        truncated = self.current_step >= self.max_episode_steps
        if self.render_mode == "human": self.render()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def get_delayed_target_buffer(self, buffer_length: int) -> np.ndarray:
        """
        Extracts the history buffer of size `buffer_length` accounting for current delay.
        Returns flattened array of shape (buffer_length * 15,).
        """
        history_len = len(self.leader_q_history)
        delay_steps = self.delay_simulator.get_state_delay_steps(history_len)
        
        # Normalize delay for input feature
        normalized_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        
        # Calculate indices
        most_recent_idx = -1 - delay_steps
        
        buffer_seq = []
        # Loop to grab exactly 'buffer_length' steps ending at 'most_recent_idx'
        for i in range(-buffer_length + 1, 1): 
            idx = np.clip(most_recent_idx + i, -history_len, -1)
            
            # Construct 15D vector [q, qd, delay]
            step_vector = self._normalize_input(
                self.leader_q_history[idx], 
                self.leader_qd_history[idx], 
                normalized_delay
            )
            buffer_seq.append(step_vector)
            
        return np.array(buffer_seq).flatten().astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the observation vector for Joint Training.
        Structure: [Remote_State (14D), Flattened_History_Buffer (150*15)]
        """
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        
        # Update remote history (optional, kept if you want to use it later)
        self.remote_q_history.append(remote_q.copy())
        self.remote_qd_history.append(remote_qd.copy())
        
        # 1. Get Delayed History (The "Eye" Input)
        history_flat = self.get_delayed_target_buffer(cfg.RNN_SEQUENCE_LENGTH)
                
        obs = np.concatenate([
            remote_q, remote_qd, 
            history_flat
        ]).astype(np.float32)
        
        return obs

    def get_true_current_target(self) -> np.ndarray:
        if not self.leader_q_history:
            return np.concatenate([self.initial_qpos, np.zeros(self.n_joints)])
        return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])

    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, float]:
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()
        
        pos_err = true_target[:self.n_joints] - remote_q
        vel_err = true_target[self.n_joints:] - remote_qd
        
        # Position Tracking (Stricter penalty)
        r_pos = -cfg.TRACKING_ERROR_SCALE * np.sum(np.abs(pos_err))
        
        # Velocity Tracking
        r_vel = -cfg.VELOCITY_ERROR_SCALE * np.sum(vel_err**2)
        
        total_reward = r_pos + r_vel
        return float(total_reward), float(total_reward)

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
        true_state = self.get_true_current_target() # 14D (q, qd)
        
        return {
            "prediction_error": 0.0, # Env no longer predicts
            "current_delay_steps": self._cached_delay_steps,
            "is_in_warmup": is_warmup,
            "true_state": true_state 
        }

    def render(self):
        pass
        
    def close(self):
        if self.viewer is not None: plt.close(self.viewer)