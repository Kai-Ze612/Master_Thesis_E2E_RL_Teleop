"""
Gymnasium Training environment with Autoregressive LSTM Integration.

LSTM: Output predicted q and qd
SAC: Learn to compensate torque based on predicted states

Goal: Min tracking error under delay with minimal torque compensation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from collections import deque
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import os

from End_to_End_RL_In_Teleoperation.agent.local_robot_simulator import LocalRobotSimulator, TrajectoryType
from End_to_End_RL_In_Teleoperation.agent.remote_robot_simulator import RemoteRobotSimulator
from End_to_End_RL_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import End_to_End_RL_In_Teleoperation.config.robot_config as cfg

class TeleoperationEnvWithDelay(gym.Env):

    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': cfg.DEFAULT_CONTROL_FREQ}
    
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.LOW_DELAY,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_trajectory: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        lstm_model_path: Optional[str] = None, # Not used in E2E Env
    ):
        super().__init__()
        
        # Rendering setup
        self.render_mode = render_mode
        self.viewer = None
        self._step_counter = 0
        
        # RL/Env parameters
        self.max_episode_steps = cfg.MAX_EPISODE_STEPS
        self.control_freq = cfg.DEFAULT_CONTROL_FREQ
        self.current_step = 0
        self.episode_count = 0
        
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
        self.remote_robot = RemoteRobotSimulator(delay_config=delay_config, seed=seed)
        
        # Buffers
        self.leader_q_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.remote_q_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        self.remote_qd_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        
        # Warmup Logic
        self.warmup_time = cfg.WARM_UP_DURATION 
        self.warmup_steps = int(self.warmup_time * self.control_freq)
        self.steps_remaining_in_warmup = 0
        self.grace_period_steps = self.warmup_steps + int(cfg.NO_DELAY_DURATION * self.control_freq)
        
        # Gym Spaces 
        self.action_space = spaces.Box(
            low=-cfg.MAX_TORQUE_COMPENSATION,
            high=cfg.MAX_TORQUE_COMPENSATION, 
            shape=(self.n_joints,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.OBS_DIM,), dtype=np.float32)

        # Internal State
        self.last_target_q: Optional[np.ndarray] = None 
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1
        
        leader_start_q, _ = self.leader.reset(seed=seed)
        self.remote_robot.reset(initial_qpos=self.initial_qpos)
        
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        self.remote_q_history.clear()
        self.remote_qd_history.clear()
        
        prefill_count = cfg.RNN_SEQUENCE_LENGTH + 20
        for _ in range(prefill_count):
            self.leader_q_history.append(leader_start_q.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))
            
        start_target_q = leader_start_q.copy()
        for _ in range(cfg.REMOTE_HISTORY_LEN):
            self.remote_q_history.append(start_target_q.copy())
            self.remote_qd_history.append(np.zeros(self.n_joints))
            
        self.steps_remaining_in_warmup = self.warmup_steps
        self.last_target_q = start_target_q.copy()
        
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_counter += 1
        self.current_step += 1
        
        # Leader step
        new_leader_q, new_leader_qd, _, _, _, _ = self.leader.step()
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())

        # NOTE: In E2E training, the Environment does NOT predict. 
        # The 'action' comes from the Agent which already used the prediction.
        # However, for the simulation step, we need a "Target" for the PD controller.
        
        # For simplicity in E2E Env: 
        # The PD target is simply the "Delayed" ground truth + Agent's knowledge (implied).
        # OR: We assume the agent's action (torque) is enough.
        # BUT: The remote robot needs a position target for its internal PD.
        
        # LOGIC CHANGE: We use the Delayed Leader State as the "Naive Target" for the PD controller.
        # The Agent provides torque to fix the error.
        current_delayed_target = self._get_delayed_q_now()
        
        # Warmup handling
        if self.steps_remaining_in_warmup > 0:
            self.steps_remaining_in_warmup -= 1
            safe_target_q = self.initial_qpos.copy()
            torque_compensation = np.zeros(self.n_joints)
            target_qd = np.zeros(self.n_joints)
        else:
            # Use delayed input as base target
            raw_target_q = current_delayed_target[:self.n_joints]
            target_qd = current_delayed_target[self.n_joints:]
            
            # Safety Clamp
            if self.last_target_q is None: self.last_target_q = self.remote_robot.get_joint_state()[0]
            delta_q = raw_target_q - self.last_target_q
            clamped_delta_q = np.clip(delta_q, -cfg.MAX_JOINT_CHANGE_PER_STEP, cfg.MAX_JOINT_CHANGE_PER_STEP)
            safe_target_q = self.last_target_q + clamped_delta_q
            self.last_target_q = safe_target_q.copy()
            
            torque_compensation = action

        true_target_full = self.get_true_current_target()
        true_pos = true_target_full[:self.n_joints]
        
        # Step Remote
        self.remote_robot.step(safe_target_q, target_qd, torque_compensation, true_local_q=true_pos)
        
        # Reward
        reward, _ = self._calculate_reward(action)
        
        remote_q, _ = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()
        joint_error_norm = np.linalg.norm(true_target[:self.n_joints] - remote_q)
        
        # Check Termination (Modified: No prediction error check)
        terminated, term_penalty = self._check_termination(joint_error_norm, remote_q)
        if terminated: reward += term_penalty
        
        truncated = self.current_step >= self.max_episode_steps
        if self.render_mode == "human": self.render()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_delayed_q_now(self) -> np.ndarray:
        """Returns the current delayed observation (Simple, no prediction)."""
        history_len = len(self.leader_q_history)
        if self.current_step < self.grace_period_steps:
             return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])
        
        delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
        idx = -1 - delay_steps
        # Safety check
        idx = max(-history_len, idx)
        return np.concatenate([self.leader_q_history[idx], self.leader_qd_history[idx]])

    def _get_observation(self) -> np.ndarray:
        """
        Constructs observation.
        CRITICAL: The 'Prediction' (84-98) and 'Error' (98-112) fields are filled with
        ZEROS (or simple delayed state). The SACTrainer will PATCH this with the live prediction.
        """
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        self.remote_q_history.append(remote_q.copy())
        self.remote_qd_history.append(remote_qd.copy())
        
        # Placeholder for prediction (Trainer fills this)
        pred_q = np.zeros(self.n_joints) 
        pred_qd = np.zeros(self.n_joints)
        error_q = np.zeros(self.n_joints)
        error_qd = np.zeros(self.n_joints)
        
        rem_q_hist = np.concatenate(list(self.remote_q_history))
        rem_qd_hist = np.concatenate(list(self.remote_qd_history))
        
        hist_len = len(self.leader_q_history)
        d_steps = 0 if self.current_step < self.grace_period_steps else self.delay_simulator.get_observation_delay_steps(hist_len)
        norm_delay = float(d_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        
        obs = np.concatenate([
            remote_q, remote_qd, 
            rem_q_hist, rem_qd_hist,
            pred_q, pred_qd, # Placeholders
            error_q, error_qd, # Placeholders
            [norm_delay]
        ]).astype(np.float32)
        
        return obs

    def get_true_current_target(self) -> np.ndarray:
        if not self.leader_q_history:
            return np.concatenate([self.initial_qpos, np.zeros(self.n_joints)])
        return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])

    def get_delayed_target_buffer(self, buffer_length: int) -> np.ndarray:
        """
        Public method used by SACTrainer to fetch the raw sequence input.
        """
        history_len = len(self.leader_q_history)
        if history_len == 0:
             init_vec = np.concatenate([self.initial_qpos, np.zeros(self.n_joints), [0.0]])
             return np.tile(init_vec, (buffer_length, 1)).flatten().astype(np.float32)

        delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
        normalized_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        most_recent_idx = -1 - delay_steps
        
        buffer_seq = []
        for i in range(-buffer_length + 1, 1): 
            target_history_idx = most_recent_idx + i 
            safe_idx = np.clip(target_history_idx, -history_len, -1)
            step_vector = np.concatenate([
                self.leader_q_history[safe_idx],
                self.leader_qd_history[safe_idx],
                [normalized_delay]
            ])
            buffer_seq.append(step_vector)
            
        return np.array(buffer_seq).flatten().astype(np.float32)

    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, float]:
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()
        
        pos_err = true_target[:self.n_joints] - remote_q
        vel_err = true_target[self.n_joints:] - remote_qd
       
        r_pos = -cfg.TRACKING_ERROR_SCALE * np.sum(pos_err**2)
        r_vel = -cfg.VELOCITY_ERROR_SCALE * np.sum(vel_err**2)  
        r_tracking = r_pos + r_vel
        r_action = -cfg.ACTION_PENALTY_WEIGHT * np.mean(action**2)
        return float(r_tracking + r_action), float(r_tracking)

    def _check_termination(self, joint_error: float, remote_q: np.ndarray) -> Tuple[bool, float]:
        # Removed Prediction Error Check
        if not np.all(np.isfinite(remote_q)): return True, -100.0
        
        at_limits = (np.any(remote_q <= self.joint_limits_lower + cfg.JOINT_LIMIT_MARGIN) or 
                    np.any(remote_q >= self.joint_limits_upper - cfg.JOINT_LIMIT_MARGIN))
        
        high_error = joint_error > self.max_joint_error
        
        terminated = at_limits or high_error 
        penalty = -10.0 if terminated else 0.0
        return terminated, penalty

    def _get_info(self):
        phase = "deployment" if self.current_step >= self.grace_period_steps else "warmup"
        return {
            "is_in_warmup": (phase == "warmup"),
            "phase": phase
        }

    def render(self): pass
    def close(self):
        if self.viewer is not None: plt.close(self.viewer)