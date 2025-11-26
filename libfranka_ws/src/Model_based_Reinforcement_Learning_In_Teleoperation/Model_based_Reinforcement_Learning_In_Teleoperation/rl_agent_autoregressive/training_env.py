"""
Gymnasium Training environment with Autoregressive LSTM Integration.

Pipeline:
1. LocalRobotSimulator: Generates ground truth trajectory.
2. DelaySimulator: Delays the ground truth data.
3. StateEstimator (Internal): Performs Autoregressive Rollout on delayed data to predict T_now.
4. Observation: Assembled from RemoteState + PredictedState + History.
5. RL Agent: Receives Observation, outputs Torque (7D).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from collections import deque
from typing import Tuple, Dict, Any, Optional
import warnings
import matplotlib.pyplot as plt
import os

# Custom imports
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import LocalRobotSimulator, TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.remote_robot_simulator import RemoteRobotSimulator
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator 

# Configuration imports
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

class TeleoperationEnvWithDelay(gym.Env):

    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': cfg.DEFAULT_CONTROL_FREQ}
    
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.LOW_DELAY,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_trajectory: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        lstm_model_path: Optional[str] = cfg.LSTM_MODEL_PATH, # Changed to Optional
    ):
        super().__init__()
        
        # Device setup for LSTM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        # Rendering setup
        self.render_mode = render_mode
        self.viewer = None
        self.ax = None
        self.plot_history_len = 1000
        self.hist_tracking_reward = deque(maxlen=self.plot_history_len)
        self.hist_total_reward = deque(maxlen=self.plot_history_len)
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
        
        # Warmup / Phase Logic
        self.warmup_time = cfg.WARM_UP_DURATION 
        self.warmup_steps = int(self.warmup_time * self.control_freq)
        self.steps_remaining_in_warmup = 0
        self.grace_period_steps = int(cfg.NO_DELAY_DURATION * self.control_freq)

        # --- LSTM State Estimator Setup ---
        self.lstm = None
        if lstm_model_path is not None:
            self._load_lstm_model(lstm_model_path)
        else:
            # print("Env initialized WITHOUT LSTM model (Passthrough Mode).")
            pass

        self._last_predicted_target: Optional[np.ndarray] = None
        self.max_ar_steps = 50 
        
        # --- Action Space (TORQUE ONLY) ---
        self.action_space = spaces.Box(
            low=-cfg.MAX_TORQUE_COMPENSATION,
            high=cfg.MAX_TORQUE_COMPENSATION, 
            shape=(self.n_joints,), 
            dtype=np.float32
        )
        
        # --- Observation Space ---
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.OBS_DIM,), dtype=np.float32)
        
        self.last_target_q: Optional[np.ndarray] = None 

    def _load_lstm_model(self, path):
        """Loads the frozen Autoregressive LSTM."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"LSTM model not found at {path}")
            
        try:
            self.lstm = StateEstimator().to(self.device)
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            if 'state_estimator_state_dict' in checkpoint:
                self.lstm.load_state_dict(checkpoint['state_estimator_state_dict'])
            else:
                self.lstm.load_state_dict(checkpoint)
                
            self.lstm.eval()
            for param in self.lstm.parameters():
                param.requires_grad = False
        except Exception as e:
            raise RuntimeError(f"Failed to load LSTM in Environment: {e}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1
        self._step_counter = 0
        self._last_predicted_target = None
        
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
            
        self.steps_remaining_in_warmup = 0
        self.last_target_q = start_target_q.copy()
        
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_counter += 1
        self.current_step += 1
        
        # 1. Step Leader
        new_leader_q, new_leader_qd, _, _, _, _ = self.leader.step()
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())

        # 2. LSTM Inference
        current_predicted_target = self._perform_ar_prediction_step()
        self._last_predicted_target = current_predicted_target

        if self.steps_remaining_in_warmup > 0:
            self.steps_remaining_in_warmup -= 1
            safe_target_q = self._get_delayed_q()
            torque_compensation = np.zeros(self.n_joints)
        else:
            raw_target_q = current_predicted_target[:self.n_joints]
            
            if self.last_target_q is None:
                self.last_target_q = self.remote_robot.get_joint_state()[0]

            delta_q = raw_target_q - self.last_target_q
            clamped_delta_q = np.clip(delta_q, -cfg.MAX_JOINT_CHANGE_PER_STEP, cfg.MAX_JOINT_CHANGE_PER_STEP)
            safe_target_q = self.last_target_q + clamped_delta_q
            self.last_target_q = safe_target_q.copy()
            
            torque_compensation = action

        # 3. Step Remote
        target_qd = current_predicted_target[self.n_joints:]
        self.remote_robot.step(safe_target_q, target_qd, torque_compensation)
        
        # 4. Reward & Term
        reward, r_tracking = self._calculate_reward(action)
        self.hist_total_reward.append(reward)
        self.hist_tracking_reward.append(r_tracking)
        
        remote_q, _ = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()
        
        joint_error_norm = np.linalg.norm(true_target[:self.n_joints] - remote_q)
        terminated, term_penalty = self._check_termination(joint_error_norm, remote_q)
        if terminated: reward += term_penalty
        
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human": self.render()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_delayed_q(self) -> np.ndarray:
        delay = self.delay_simulator.get_observation_delay_steps(len(self.leader_q_history))
        idx = -1 - delay
        return self.leader_q_history[idx]

    def _perform_ar_prediction_step(self) -> np.ndarray:
        """
        Simulates the Autoregressive Loop with correct scaling and delay updates.
        """
        # --- PASSTHROUGH LOGIC ---
        delay = self.delay_simulator.get_observation_delay_steps(len(self.leader_q_history))
        idx = -1 - delay
        delayed_state = np.concatenate([self.leader_q_history[idx], self.leader_qd_history[idx]])
        
        if self.lstm is None:
            return delayed_state
        # -------------------------

        history_len = len(self.leader_q_history)
        
        if self.current_step < self.grace_period_steps:
            delay_steps = 0
        else:
            delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
            
        if delay_steps == 0:
            return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])
            
        normalized_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        most_recent_idx = -1 - delay_steps
        
        # Build initial sequence
        seq_buffer = []
        start_idx = most_recent_idx - cfg.RNN_SEQUENCE_LENGTH + 1
        
        for i in range(start_idx, most_recent_idx + 1):
            idx = max(-len(self.leader_q_history), i)
            step_vec = np.concatenate([
                self.leader_q_history[idx],
                self.leader_qd_history[idx],
                [normalized_delay]
            ])
            seq_buffer.append(step_vec)
            
        input_tensor = torch.tensor(np.array(seq_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
            
        steps_to_run = min(delay_steps, self.max_ar_steps)
        
        # FIX 1: Initialize hidden state from the sequence
        with torch.no_grad():
            _, hidden_state = self.lstm.lstm(input_tensor)
        
        # FIX 2: Start AR loop from last observation
        last_obs = input_tensor[0, -1, :]
        curr_q = last_obs[:self.n_joints].clone()
        curr_qd = last_obs[self.n_joints:2*self.n_joints].clone()
        current_delay_norm = normalized_delay
        
        dt = 1.0 / self.control_freq  # DT in seconds
        
        with torch.no_grad():
            for _ in range(steps_to_run):
                # Build current input
                delay_t = torch.tensor([current_delay_norm], device=self.device)
                current_input = torch.cat([curr_q, curr_qd, delay_t], dim=0).view(1, 1, -1)
                
                # FIX 3: Use forward_step with hidden state
                residual_t, hidden_state = self.lstm.forward_step(current_input, hidden_state)
                
                # FIX 4: Correct scaling - MULTIPLY by TARGET_DELTA_SCALE (matching training)
                residual = residual_t[0] * cfg.TARGET_DELTA_SCALE
                
                curr_q = curr_q + residual[:self.n_joints]
                curr_qd = curr_qd + residual[self.n_joints:]
                
                # FIX 5: Update delay (time since observation increases as we predict forward)
                current_delay_norm = current_delay_norm + (dt / cfg.DELAY_INPUT_NORM_FACTOR)
                
        return np.concatenate([curr_q.cpu().numpy(), curr_qd.cpu().numpy()])

    def _get_observation(self) -> np.ndarray:
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        self.remote_q_history.append(remote_q.copy())
        self.remote_qd_history.append(remote_qd.copy())
        
        if self._last_predicted_target is None:
             self._last_predicted_target = self._perform_ar_prediction_step()
             
        pred_q = self._last_predicted_target[:self.n_joints]
        pred_qd = self._last_predicted_target[self.n_joints:]
        
        error_q = pred_q - remote_q
        error_qd = pred_qd - remote_qd
        
        rem_q_hist = np.concatenate(list(self.remote_q_history))
        rem_qd_hist = np.concatenate(list(self.remote_qd_history))
        
        hist_len = len(self.leader_q_history)
        d_steps = 0 if self.current_step < self.grace_period_steps else self.delay_simulator.get_observation_delay_steps(hist_len)
        norm_delay = float(d_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        
        obs = np.concatenate([
            remote_q, remote_qd, 
            rem_q_hist, rem_qd_hist,
            pred_q, pred_qd,
            error_q, error_qd,
            [norm_delay]
        ]).astype(np.float32)
        
        return obs

    def get_true_current_target(self) -> np.ndarray:
        if not self.leader_q_history:
            return np.concatenate([self.initial_qpos, np.zeros(self.n_joints)])
        return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])

    def get_delayed_target_buffer(self, buffer_length: int) -> np.ndarray:
        """Helper for LSTM pre-training data collection."""
        history_len = len(self.leader_q_history)
        if history_len == 0:
             # Just return initial state padded
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
        if not np.all(np.isfinite(remote_q)): return True, -100.0
        at_limits = (np.any(remote_q <= self.joint_limits_lower + cfg.JOINT_LIMIT_MARGIN) or 
                     np.any(remote_q >= self.joint_limits_upper - cfg.JOINT_LIMIT_MARGIN))
        high_error = joint_error > self.max_joint_error
        terminated = at_limits or high_error
        penalty = -10.0 if terminated else 0.0
        return terminated, penalty

    def _get_info(self) -> Dict[str, Any]:
        return {}

    def render(self):
        pass
    def close(self):
        if self.viewer is not None: plt.close(self.viewer)