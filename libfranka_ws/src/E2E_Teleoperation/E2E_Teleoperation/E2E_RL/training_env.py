"""
RL Training Environment with Delays
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import mujoco
from collections import deque
from typing import Tuple, Dict, Any, Optional

from E2E_Teleoperation.E2E_RL.local_robot_simulator import LocalRobotSimulator, TrajectoryType
from E2E_Teleoperation.E2E_RL.remote_robot_simulator import RemoteRobotSimulator
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import E2E_Teleoperation.config.robot_config as cfg


class TeleoperationEnv(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': cfg.CONTROL_FREQ}
    
    def __init__(
        self, 
        delay_config=ExperimentConfig.HIGH_VARIANCE, 
        trajectory_type=TrajectoryType.FIGURE_8, 
        randomize_trajectory=False, 
        seed=None, 
        render_mode=None
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.render_mode = render_mode
        self.max_episode_steps = cfg.MAX_EPISODE_STEPS
        
        # 1. Simulators
        self.delay_simulator = DelaySimulator(cfg.CONTROL_FREQ, config=delay_config, seed=seed)
        self.leader = LocalRobotSimulator(
            trajectory_type=trajectory_type, 
            randomize_params=randomize_trajectory
        )
        should_render = (self.render_mode == "human")
        self.remote = RemoteRobotSimulator(
            delay_config=delay_config, 
            seed=seed, 
            render=should_render
        )
        
        # 2. Teacher (Inverse Dynamics) Setup
        # We reuse the remote robot's model structure for math
        self._teacher_model = self.remote.model
        self._teacher_data = mujoco.MjData(self._teacher_model)
        self._prev_total_torque = np.zeros(cfg.N_JOINTS)
        
        # 3. Buffers
        self.leader_hist = deque(maxlen=200) # For delay calculation reference
        self.remote_hist_q = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        self.remote_hist_qd = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        
        # 4. Action & Observation Spaces
        self.action_space = spaces.Box(
            low=-cfg.MAX_ACTION_TORQUE, 
            high=cfg.MAX_ACTION_TORQUE, 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(cfg.OBS_DIM,), 
            dtype=np.float32
        )
        
        # State
        self._predicted_state = None
        self.step_count = 0
        self.initial_qpos = cfg.INITIAL_JOINT_CONFIG.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._predicted_state = None
        self._prev_total_torque = np.zeros(cfg.N_JOINTS)
        
        # Reset Robots
        l_q, _ = self.leader.reset(seed=seed)
        self.remote.reset(initial_qpos=self.initial_qpos)
        
        # Clear & Fill History
        self.leader_hist.clear()
        self.remote_hist_q.clear()
        self.remote_hist_qd.clear()
        
        # Pre-fill history with initial state
        init_state = (l_q, np.zeros(cfg.N_JOINTS))
        for _ in range(cfg.RNN_SEQUENCE_LENGTH + 50):
            self.leader_hist.append(init_state)
            self.remote_hist_q.append(self.initial_qpos.copy())
            self.remote_hist_qd.append(np.zeros(cfg.N_JOINTS))
            
        return self._get_obs(), {}

    def set_predicted_state(self, predicted_state):
        """Allows the agent to inject its state prediction for logging/metrics."""
        self._predicted_state = predicted_state

    def step(self, action):
        self.step_count += 1
        
        # 1. Step Leader (Local Robot)
        l_q, l_qd, _, _, _, _ = self.leader.step()
        self.leader_hist.append((l_q.copy(), l_qd.copy()))
        
        # 2. Get Remote State
        r_q, r_qd = self.remote.get_joint_state()
        self.remote_hist_q.append(r_q)
        self.remote_hist_qd.append(r_qd)
        
        # 3. Calculate Teacher Action (Ground Truth Label)
        # We always compute this so we can store it in the buffer for BC
        target_q, target_qd = self.leader_hist[-1] # Current True Target
        
        # This computes ID + Gravity + PD (The ideal total torque)
        teacher_total_torque = self._compute_teacher_torque(r_q, r_qd, target_q, target_qd)
        
        # 4. Apply Action
        # In Direct Torque Control, the action IS the total torque.
        # We clip it to physical limits to be safe.
        applied_torque = np.clip(action, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
        
        # Step Physics
        self.remote.step(
            target_q=target_q, 
            target_qd=target_qd, 
            torque_input=applied_torque,
            true_local_q=target_q,
            predicted_q=self._predicted_state
        )
        
        # 5. Reward (Tracking Accuracy)
        dist = np.linalg.norm(target_q - r_q)
        reward = np.exp(-5.0 * dist)
        
        terminated = dist > cfg.MAX_JOINT_ERROR_TERMINATION
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            "teacher_action": teacher_total_torque, # LABEL for BC
            "true_state": np.concatenate([target_q, target_qd]), # LABEL for Encoder
            "tracking_error": dist
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _compute_teacher_torque(self, curr_q, curr_qd, des_q, des_qd):
        """
        Computes the Total Ideal Torque (ID + Gravity + PD) to reach the target.
        This serves as the 'Expert' label for Behavioral Cloning.
        """
        # 1. PD Control (Desired Acceleration)
        kp, kd = cfg.TEACHER_KP, cfg.TEACHER_KD
        qdd_des = kp * (des_q - curr_q) + kd * (des_qd - curr_qd)
        
        # 2. Inverse Dynamics
        self._teacher_data.qpos[:7] = curr_q
        self._teacher_data.qvel[:7] = curr_qd
        self._teacher_data.qacc[:7] = qdd_des
        mujoco.mj_inverse(self._teacher_model, self._teacher_data)
        
        # qfrc_inverse includes Gravity + Coriolis + Inertial forces
        raw_torque = self._teacher_data.qfrc_inverse[:7].copy()
        
        # 3. Smoothing (Optional, to avoid jerky labels)
        alpha = cfg.TEACHER_SMOOTHING
        smoothed_torque = (1 - alpha) * raw_torque + alpha * self._prev_total_torque
        self._prev_total_torque = smoothed_torque
        
        return smoothed_torque

    def _get_obs(self):
        # 1. Remote State (Normalized)
        r_q, r_qd = self.remote_hist_q[-1], self.remote_hist_qd[-1]
        state_norm = np.concatenate([(r_q - cfg.Q_MEAN)/cfg.Q_STD, (r_qd - cfg.QD_MEAN)/cfg.QD_STD])
        
        # 2. Remote History (Normalized)
        hist_seq = []
        for i in range(cfg.RNN_SEQUENCE_LENGTH):
            idx = i - cfg.RNN_SEQUENCE_LENGTH
            q = (self.remote_hist_q[i] - cfg.Q_MEAN)/cfg.Q_STD
            qd = (self.remote_hist_qd[i] - cfg.QD_MEAN)/cfg.QD_STD
            hist_seq.extend(np.concatenate([q, qd]))
            
        # 3. Delayed Target History (Normalized)
        delay_steps = self.delay_simulator.get_state_delay_steps(len(self.leader_hist))
        norm_delay = delay_steps / cfg.DELAY_INPUT_NORM_FACTOR
        
        target_seq = []
        # Calculate start index in leader history to simulate delay
        # We need a window of RNN_SEQ_LEN ending at (Now - Delay)
        end_idx = len(self.leader_hist) - 1 - delay_steps
        start_idx = end_idx - cfg.RNN_SEQUENCE_LENGTH + 1
        
        # Handle boundary (if delay > history length, though history buffer is large)
        start_idx = max(0, start_idx)
        
        # Reconstruct sequence
        for i in range(cfg.RNN_SEQUENCE_LENGTH):
            curr_idx = start_idx + i
            if curr_idx < len(self.leader_hist) and curr_idx >= 0:
                q, qd = self.leader_hist[curr_idx]
            else:
                # Pad with oldest available or zeros if empty (shouldn't happen due to reset)
                q, qd = self.leader_hist[0]
                
            q_norm = (q - cfg.Q_MEAN)/cfg.Q_STD
            qd_norm = (qd - cfg.QD_MEAN)/cfg.QD_STD
            target_seq.extend(np.concatenate([q_norm, qd_norm, [norm_delay]]))
            
        return np.concatenate([state_norm, hist_seq, target_seq], dtype=np.float32)

    def render(self):
        # Render is handled inside RemoteRobotSimulator
        pass
    
    def close(self):
        self.remote.close()