"""
Create RL Training Environment with Delays
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
    """
    Teleoperation environment with realistic observation timing.
    
    Key Features:
    1. Simulates sporadic observation arrivals based on network delay
    2. Returns `has_new_obs` flag in info dict
    3. Tracks observation arrival timing for proper LSTM training
    """
    
    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': cfg.CONTROL_FREQ}
    
    def __init__(
        self,
        delay_config=ExperimentConfig.HIGH_VARIANCE,
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_trajectory=False,
        seed=None,
        render_mode=None,
        simulate_obs_timing: bool = True  # NEW: Enable realistic obs timing
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.render_mode = render_mode
        self.max_episode_steps = cfg.MAX_EPISODE_STEPS
        self.simulate_obs_timing = simulate_obs_timing
        
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
            render=should_render,
            verbose=False
        )
        
        # 2. Teacher (Inverse Dynamics) Setup
        self._teacher_model = self.remote.model
        self._teacher_data = mujoco.MjData(self._teacher_model)
        self._prev_total_torque = np.zeros(cfg.N_JOINTS)
        
        # 3. Buffers
        self.leader_hist = deque(maxlen=200)
        self.remote_hist_q = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        self.remote_hist_qd = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        
        # 4. Observation Timing State (NEW)
        self._obs_send_queue = []      # Queue of (send_time, obs_data)
        self._last_received_obs = None  # Last received delayed observation
        self._steps_since_obs = 0       # Counter for AR steps
        self._current_delay_steps = 0   # Current delay in steps
        
        # 5. Action & Observation Spaces
        self.action_space = spaces.Box(
            low=-cfg.MAX_ACTION_TORQUE,
            high=cfg.MAX_ACTION_TORQUE,
            shape=(cfg.N_JOINTS,),
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
        
        # Statistics
        self._total_obs_arrivals = 0
        self._total_ar_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._predicted_state = None
        self._prev_total_torque = np.zeros(cfg.N_JOINTS)
        
        # Reset observation timing
        self._obs_send_queue = []
        self._last_received_obs = None
        self._steps_since_obs = 0
        self._current_delay_steps = self.delay_simulator.get_state_delay_steps(100)
        self._total_obs_arrivals = 0
        self._total_ar_steps = 0
        
        # Reset robots
        l_q, _ = self.leader.reset(seed=seed)
        self.remote.reset(initial_qpos=self.initial_qpos)
        
        # Clear & Fill history
        self.leader_hist.clear()
        self.remote_hist_q.clear()
        self.remote_hist_qd.clear()
        
        init_state = (l_q.copy(), np.zeros(cfg.N_JOINTS))
        for _ in range(cfg.RNN_SEQUENCE_LENGTH + 50):
            self.leader_hist.append(init_state)
            self.remote_hist_q.append(self.initial_qpos.copy())
            self.remote_hist_qd.append(np.zeros(cfg.N_JOINTS))
        
        # Initialize observation queue with initial observations
        # Simulate that observations were sent in the past
        for i in range(cfg.RNN_SEQUENCE_LENGTH):
            arrival_time = i  # They arrive at the start
            obs_data = (l_q.copy(), np.zeros(cfg.N_JOINTS))
            self._obs_send_queue.append((arrival_time, obs_data))
        
        # First step always has observation
        self._last_received_obs = self._get_obs_sequence()
        
        return self._get_obs(), {'has_new_obs': True, 'steps_since_obs': 0}

    def set_predicted_state(self, predicted_state):
        """Allows the agent to inject its state prediction for logging."""
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
        
        # 3. Simulate Observation Timing (NEW)
        has_new_obs = self._simulate_observation_arrival()
        
        if has_new_obs:
            self._steps_since_obs = 0
            self._total_obs_arrivals += 1
        else:
            self._steps_since_obs += 1
            self._total_ar_steps += 1
        
        # 4. Calculate Teacher Action
        target_q, target_qd = self.leader_hist[-1]
        teacher_total_torque = self._compute_teacher_torque(r_q, r_qd, target_q, target_qd)
        
        # 5. Apply Action
        applied_torque = np.clip(action, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
        
        self.remote.step(
            target_q=target_q,
            target_qd=target_qd,
            torque_input=applied_torque,
            true_local_q=target_q,
            predicted_q=self._predicted_state
        )
        
        # 6. Reward
        r_q_new, _ = self.remote.get_joint_state()
        dist = np.linalg.norm(target_q - r_q_new)
        reward = np.exp(-5.0 * dist)
        
        terminated = dist > cfg.MAX_JOINT_ERROR_TERMINATION
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'teacher_action': teacher_total_torque,
            'true_state': np.concatenate([target_q, target_qd]),
            'tracking_error': dist,
            'has_new_obs': has_new_obs,           # NEW
            'steps_since_obs': self._steps_since_obs,  # NEW
            'current_delay_steps': self._current_delay_steps,  # NEW
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _simulate_observation_arrival(self) -> bool:
        """
        Simulate realistic observation arrival based on network delay.
        
        In real deployment:
        - Local robot sends observation at time t
        - Observation arrives at remote at time t + delay
        - Delay varies between 90-290ms (23-73 steps at 250Hz)
        
        Returns:
            has_new_obs: True if a new observation arrived this step
        """
        if not self.simulate_obs_timing:
            # Training mode without timing simulation: always have observation
            return True
        
        # Send current observation (will arrive after delay)
        current_obs = (
            self.leader_hist[-1][0].copy(),  # q
            self.leader_hist[-1][1].copy()   # qd
        )
        
        # Sample new delay
        delay_steps = self.delay_simulator.get_state_delay_steps(len(self.leader_hist))
        arrival_time = self.step_count + delay_steps
        
        self._obs_send_queue.append((arrival_time, current_obs))
        
        # Check if any observation has arrived
        has_new_obs = False
        arrived_obs = []
        
        remaining_queue = []
        for arrival_time, obs_data in self._obs_send_queue:
            if arrival_time <= self.step_count:
                arrived_obs.append((arrival_time, obs_data))
                has_new_obs = True
            else:
                remaining_queue.append((arrival_time, obs_data))
        
        self._obs_send_queue = remaining_queue
        
        # Update last received observation
        if arrived_obs:
            # Take the most recent arrival
            arrived_obs.sort(key=lambda x: x[0])
            _, latest_obs = arrived_obs[-1]
            self._last_received_obs = self._build_obs_sequence_from_arrival(latest_obs)
            self._current_delay_steps = delay_steps
        
        return has_new_obs

    def _build_obs_sequence_from_arrival(self, latest_obs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Build observation sequence from arrived observation."""
        # For simplicity, use the delayed history buffer
        # In a more sophisticated implementation, you'd track the full sequence
        return self._get_obs_sequence()

    def _compute_teacher_torque(self, curr_q, curr_qd, des_q, des_qd):
        """Compute ideal torque using inverse dynamics."""
        kp, kd = cfg.TEACHER_KP, cfg.TEACHER_KD
        qdd_des = kp * (des_q - curr_q) + kd * (des_qd - curr_qd)
        
        self._teacher_data.qpos[:7] = curr_q
        self._teacher_data.qvel[:7] = curr_qd
        self._teacher_data.qacc[:7] = qdd_des
        mujoco.mj_inverse(self._teacher_model, self._teacher_data)
        
        raw_torque = self._teacher_data.qfrc_inverse[:7].copy()
        
        alpha = cfg.TEACHER_SMOOTHING
        smoothed_torque = (1 - alpha) * raw_torque + alpha * self._prev_total_torque
        self._prev_total_torque = smoothed_torque
        
        return smoothed_torque

    def _get_obs_sequence(self) -> np.ndarray:
        """Get the delayed observation sequence for LSTM input."""
        delay_steps = self.delay_simulator.get_state_delay_steps(len(self.leader_hist))
        norm_delay = delay_steps / cfg.DELAY_INPUT_NORM_FACTOR
        
        target_seq = []
        end_idx = len(self.leader_hist) - 1 - delay_steps
        start_idx = max(0, end_idx - cfg.RNN_SEQUENCE_LENGTH + 1)
        
        for i in range(cfg.RNN_SEQUENCE_LENGTH):
            curr_idx = start_idx + i
            if 0 <= curr_idx < len(self.leader_hist):
                q, qd = self.leader_hist[curr_idx]
            else:
                q, qd = self.leader_hist[0]
            
            q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
            qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
            target_seq.extend(np.concatenate([q_norm, qd_norm, [norm_delay]]))
        
        return np.array(target_seq, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Construct full observation vector."""
        # 1. Remote State (Normalized)
        r_q, r_qd = self.remote_hist_q[-1], self.remote_hist_qd[-1]
        state_norm = np.concatenate([
            (r_q - cfg.Q_MEAN) / cfg.Q_STD,
            (r_qd - cfg.QD_MEAN) / cfg.QD_STD
        ])
        
        # 2. Remote History (Normalized)
        hist_seq = []
        for i in range(cfg.RNN_SEQUENCE_LENGTH):
            q = (self.remote_hist_q[i] - cfg.Q_MEAN) / cfg.Q_STD
            qd = (self.remote_hist_qd[i] - cfg.QD_MEAN) / cfg.QD_STD
            hist_seq.extend(np.concatenate([q, qd]))
        
        # 3. Target History (Delayed, Normalized)
        target_seq = self._get_obs_sequence()
        
        return np.concatenate([state_norm, hist_seq, target_seq], dtype=np.float32)

    def get_obs_timing_stats(self) -> Dict[str, float]:
        """Get statistics about observation timing."""
        total = self._total_obs_arrivals + self._total_ar_steps
        if total == 0:
            return {'ar_ratio': 0.0, 'obs_arrivals': 0, 'ar_steps': 0}
        
        return {
            'ar_ratio': self._total_ar_steps / total,
            'obs_arrivals': self._total_obs_arrivals,
            'ar_steps': self._total_ar_steps,
            'avg_obs_interval': total / max(1, self._total_obs_arrivals)
        }

    def render(self):
        pass

    def close(self):
        self.remote.close()