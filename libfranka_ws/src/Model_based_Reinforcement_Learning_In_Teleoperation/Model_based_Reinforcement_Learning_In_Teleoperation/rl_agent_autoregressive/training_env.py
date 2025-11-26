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
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator 

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
        lstm_model_path: str = cfg.LSTM_MODEL_PATH, 
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
        self._load_lstm_model(lstm_model_path)
        self._last_predicted_target: Optional[np.ndarray] = None
        self.max_ar_steps = 50 # Safety cap matching AgentNode
        
        # --- Action Space (TORQUE ONLY) ---
        # 7 Dimensions
        self.action_space = spaces.Box(
            low=-cfg.MAX_TORQUE_COMPENSATION,
            high=cfg.MAX_TORQUE_COMPENSATION, 
            shape=(self.n_joints,), 
            dtype=np.float32
        )
        
        # --- Observation Space ---
        # Matches OBS_DIM (112D)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.OBS_DIM,), dtype=np.float32)
        
        # State tracking
        self.last_target_q: Optional[np.ndarray] = None 

    def _load_lstm_model(self, path):
        """Loads the frozen Autoregressive LSTM."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"LSTM model not found at {path}")
            
        try:
            self.lstm = StateEstimator().to(self.device)
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Handle dictionary vs direct state_dict
            if 'state_estimator_state_dict' in checkpoint:
                self.lstm.load_state_dict(checkpoint['state_estimator_state_dict'])
            else:
                self.lstm.load_state_dict(checkpoint)
                
            self.lstm.eval() # Set to evaluation mode
            for param in self.lstm.parameters():
                param.requires_grad = False # Freeze weights
            
            # print(f"Environment loaded LSTM model from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load LSTM in Environment: {e}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1
        self._step_counter = 0
        self._last_predicted_target = None
        
        # Reset Simulators
        leader_start_q, _ = self.leader.reset(seed=seed)
        self.remote_robot.reset(initial_qpos=self.initial_qpos)
        
        # Clear Buffers
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        self.remote_q_history.clear()
        self.remote_qd_history.clear()
        
        # Pre-fill Leader Buffer (to allow LSTM inputs immediately)
        prefill_count = cfg.RNN_SEQUENCE_LENGTH + 20
        for _ in range(prefill_count):
            self.leader_q_history.append(leader_start_q.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))
            
        # Reset Remote History
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
        
        # 1. Step Leader (Ground Truth)
        new_leader_q, new_leader_qd, _, _, _, _ = self.leader.step()
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())

        # 2. Perform LSTM Inference (Autoregressive)
        # We perform prediction based on the delayed view of the leader state
        current_predicted_target = self._perform_ar_prediction_step()
        self._last_predicted_target = current_predicted_target

        if self.steps_remaining_in_warmup > 0:
            self.steps_remaining_in_warmup -= 1
            # During warmup, hold current delayed position
            safe_target_q = self._get_delayed_q()
            torque_compensation = np.zeros(self.n_joints)
        else:
            # Normal Operation
            raw_target_q = current_predicted_target[:self.n_joints]
            
            # --- Safety Ramp Logic ---
            if self.last_target_q is None:
                self.last_target_q = self.remote_robot.get_joint_state()[0]

            delta_q = raw_target_q - self.last_target_q
            clamped_delta_q = np.clip(delta_q, -cfg.MAX_JOINT_CHANGE_PER_STEP, cfg.MAX_JOINT_CHANGE_PER_STEP)
            safe_target_q = self.last_target_q + clamped_delta_q
            self.last_target_q = safe_target_q.copy()
            
            torque_compensation = action # RL Output (7D)

        # 3. Step Remote Robot
        # Use prediction velocity as feedforward
        target_qd = current_predicted_target[self.n_joints:]
        
        self.remote_robot.step(safe_target_q, target_qd, torque_compensation)
        
        # 4. Reward Calculation
        reward, r_tracking = self._calculate_reward(action)
        self.hist_total_reward.append(reward)
        self.hist_tracking_reward.append(r_tracking)
        
        # 5. Check Termination
        remote_q, _ = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()
        
        # Check physical limits
        joint_error_norm = np.linalg.norm(true_target[:self.n_joints] - remote_q)
        terminated, term_penalty = self._check_termination(joint_error_norm, remote_q)
        if terminated: reward += term_penalty
        
        truncated = self.current_step >= self.max_episode_steps

        # 6. Render
        if self.render_mode == "human": self.render()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_delayed_q(self) -> np.ndarray:
        delay = self.delay_simulator.get_observation_delay_steps(len(self.leader_q_history))
        idx = -1 - delay
        return self.leader_q_history[idx]

    def _perform_ar_prediction_step(self) -> np.ndarray:
        """
        Simulates the Autoregressive Loop found in AgentNode.
        Returns: 14D vector [pred_q, pred_qd] at t_now.
        """
        history_len = len(self.leader_q_history)
        
        # Determine Delay
        if self.current_step < self.grace_period_steps:
            delay_steps = 0
        else:
            delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
            
        normalized_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        
        # Prepare History Sequence (Input to LSTM)
        # We need seq [T_delayed - seq_len : T_delayed]
        most_recent_idx = -1 - delay_steps
        
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
            
        # Shape: (1, Seq_Len, 15)
        input_tensor = torch.tensor(np.array(seq_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Autoregressive Loop
        if delay_steps == 0:
            return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])
            
        steps_to_run = min(delay_steps, self.max_ar_steps)
        current_seq_t = input_tensor.clone()
        
        # Anchor state
        last_obs = current_seq_t[0, -1, :]
        curr_q = last_obs[:self.n_joints]
        curr_qd = last_obs[self.n_joints:2*self.n_joints]
        
        delay_tensor = torch.tensor([normalized_delay], device=self.device)
        
        with torch.no_grad():
            for _ in range(steps_to_run):
                # Predict Residual
                residual_t, _ = self.lstm(current_seq_t)
                residual = residual_t[0] / cfg.TARGET_DELTA_SCALE
                
                # Update State
                curr_q = curr_q + residual[:self.n_joints]
                curr_qd = curr_qd + residual[self.n_joints:]
                
                # Update Sequence
                new_input = torch.cat([curr_q, curr_qd, delay_tensor], dim=0).view(1, 1, -1)
                current_seq_t = torch.cat([current_seq_t[:, 1:, :], new_input], dim=1)
                
        return np.concatenate([curr_q.cpu().numpy(), curr_qd.cpu().numpy()])

    def _get_observation(self) -> np.ndarray:
        """Constructs the 112D observation."""
        
        # 1. Remote State
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        
        # 2. Update History Buffers
        self.remote_q_history.append(remote_q.copy())
        self.remote_qd_history.append(remote_qd.copy())
        
        # 3. Get Prediction
        if self._last_predicted_target is None:
             self._last_predicted_target = self._perform_ar_prediction_step()
             
        pred_q = self._last_predicted_target[:self.n_joints]
        pred_qd = self._last_predicted_target[self.n_joints:]
        
        # 4. Calc Error
        error_q = pred_q - remote_q
        error_qd = pred_qd - remote_qd
        
        # 5. Flatten History
        rem_q_hist = np.concatenate(list(self.remote_q_history))
        rem_qd_hist = np.concatenate(list(self.remote_qd_history))
        
        # 6. Delay Scalar
        hist_len = len(self.leader_q_history)
        d_steps = 0 if self.current_step < self.grace_period_steps else self.delay_simulator.get_observation_delay_steps(hist_len)
        norm_delay = float(d_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        
        # Concatenate
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
        remote_q, _ = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()[:self.n_joints]
        
        info = {
            'real_time_joint_error': np.linalg.norm(true_target - remote_q),
            'prediction_error': 0.0
        }
        
        if self._last_predicted_target is not None:
             pred_q = self._last_predicted_target[:self.n_joints]
             info['prediction_error'] = np.linalg.norm(true_target - pred_q)
             
        return info

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer, self.ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            self.viewer.suptitle(f'Live Teleoperation Tracking - Run {self.episode_count}', fontsize=12)
            self.line1, = self.ax[0].plot([], [], label='TOTAL Step Reward', color='green')
            self.ax[0].set_ylabel('Total Reward')
            self.ax[0].legend(loc='upper right')
            self.line2, = self.ax[1].plot([], [], label='Tracking Reward', color='blue')
            self.ax[1].set_ylabel('Tracking Reward')
            self.ax[1].set_xlabel('Steps')
            plt.ion()
            plt.show(block=False)

        x_data = np.arange(max(0, self._step_counter - len(self.hist_total_reward)), self._step_counter)
        self.line1.set_data(x_data, self.hist_total_reward)
        self.line2.set_data(x_data, self.hist_tracking_reward)
        for ax in self.ax:
            ax.relim()
            ax.autoscale_view()
        self.viewer.canvas.draw_idle()
        self.viewer.canvas.flush_events()
        
    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None