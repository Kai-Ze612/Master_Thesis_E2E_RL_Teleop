"""
TeleoperationEnvWithDelay - Version 11: 10x Reward

CHANGES:
1. Scaled Reward by 10.0.
   - Provides strong signal without causing Q-value explosion.
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


class TeleoperationEnvWithDelay(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': cfg.DEFAULT_CONTROL_FREQ}
    
    def __init__(self, delay_config=ExperimentConfig.HIGH_VARIANCE, trajectory_type=TrajectoryType.FIGURE_8, randomize_trajectory=False, seed=None, render_mode=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.render_mode = render_mode
        self.max_episode_steps = cfg.MAX_EPISODE_STEPS
        self.control_freq = cfg.DEFAULT_CONTROL_FREQ
        self.dt = 1.0 / self.control_freq
        self.current_step = 0
        self.n_joints = cfg.N_JOINTS
        self.initial_qpos = cfg.INITIAL_JOINT_CONFIG.copy()
        self.max_joint_error = cfg.MAX_JOINT_ERROR_TERMINATION
        self.joint_limits_lower = cfg.JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = cfg.JOINT_LIMITS_UPPER.copy()
        
        self.torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)
        self.max_acceleration = np.array([10.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0], dtype=np.float64)
        self.max_id_residual_torque = np.array([40.0, 40.0, 40.0, 40.0, 6.0, 6.0, 6.0], dtype=np.float64)
        
        self.delay_config = delay_config
        self.delay_simulator = DelaySimulator(control_freq=self.control_freq, config=delay_config, seed=seed)
        self.leader = LocalRobotSimulator(trajectory_type=trajectory_type, randomize_params=randomize_trajectory)
        should_render_remote = (self.render_mode == "human")
        self.remote_robot = RemoteRobotSimulator(delay_config=delay_config, seed=seed, render=should_render_remote)
        
        self._teacher_model = self.remote_robot.model
        self._teacher_data = mujoco.MjData(self._teacher_model)
        self._gravity_data = mujoco.MjData(self._teacher_model)
        self._teacher_kp = np.array([100.0, 100.0, 100.0, 100.0, 80.0, 60.0, 40.0], dtype=np.float64)
        self._teacher_kd = np.array([20.0, 20.0, 20.0, 20.0, 12.0, 10.0, 8.0], dtype=np.float64)
        self._prev_id_torque = np.zeros(self.n_joints)
        self._torque_smoothing_alpha = 0.3
        
        self.leader_q_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.remote_q_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        self.remote_qd_history = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        
        self._precomputed_trajectory_q = None
        self._precomputed_trajectory_qd = None
        self._current_predicted_state = None
        self.warmup_steps = int(cfg.WARM_UP_DURATION * self.control_freq)
        self.steps_remaining_in_warmup = 0
        self.grace_period_steps = self.warmup_steps + int(cfg.NO_DELAY_DURATION * self.control_freq)
        self._use_teacher = False
        self._use_ground_truth = True
        
        max_control_torque = np.array([50.0, 50.0, 50.0, 50.0, 8.0, 8.0, 8.0], dtype=np.float32)
        self.action_space = spaces.Box(low=-max_control_torque, high=max_control_torque, shape=(self.n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.OBS_DIM,), dtype=np.float32)
        
        self._cached_prediction_error = 0.0
        self._cached_delay_steps = 0
        self._last_gravity_torque = np.zeros(self.n_joints)
        self._last_id_torque = np.zeros(self.n_joints)
        self._last_teacher_action = np.zeros(self.n_joints)
        
    def set_teacher_mode(self, use_teacher: bool): self._use_teacher = use_teacher
    def set_ground_truth_mode(self, use_ground_truth: bool): self._use_ground_truth = use_ground_truth
    def _normalize_state(self, q, qd):
        q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
        qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
        return np.concatenate([q_norm, qd_norm])
    def _denormalize_q(self, q_norm): return (q_norm * cfg.Q_STD) + cfg.Q_MEAN
    def _normalize_input(self, q, qd, delay_scalar):
        state_norm = self._normalize_state(q, qd)
        return np.concatenate([state_norm, [delay_scalar]])
    def _compute_gravity_compensation(self, q):
        self._gravity_data.qpos[:self.n_joints] = q
        self._gravity_data.qvel[:self.n_joints] = 0.0
        self._gravity_data.qacc[:self.n_joints] = 0.0
        mujoco.mj_inverse(self._teacher_model, self._gravity_data)
        return self._gravity_data.qfrc_inverse[:self.n_joints].copy()
    def _compute_teacher_action(self, current_q, current_qd, target_q, target_qd):
        pos_error = target_q - current_q
        vel_error = target_qd - current_qd
        qdd_desired = self._teacher_kp * pos_error + self._teacher_kd * vel_error
        qdd_desired = np.clip(qdd_desired, -self.max_acceleration, self.max_acceleration)
        self._teacher_data.qpos[:self.n_joints] = current_q
        self._teacher_data.qvel[:self.n_joints] = current_qd
        self._teacher_data.qacc[:self.n_joints] = qdd_desired
        mujoco.mj_inverse(self._teacher_model, self._teacher_data)
        full_torque = self._teacher_data.qfrc_inverse[:self.n_joints].copy()
        gravity_torque = self._compute_gravity_compensation(current_q)
        residual_torque = full_torque - gravity_torque
        residual_torque = np.clip(residual_torque, -self.max_id_residual_torque, self.max_id_residual_torque)
        smoothed_torque = (1 - self._torque_smoothing_alpha) * residual_torque + self._torque_smoothing_alpha * self._prev_id_torque
        self._prev_id_torque = smoothed_torque.copy()
        return smoothed_torque
    def set_predicted_state(self, predicted_state): self._current_predicted_state = predicted_state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._current_predicted_state = None
        self._cached_prediction_error = 0.0
        self._cached_delay_steps = 0
        leader_start_q, _ = self.leader.reset(seed=seed)
        self._precompute_trajectory(leader_start_q)
        self.remote_robot.reset(initial_qpos=self.initial_qpos)
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        self.remote_q_history.clear()
        self.remote_qd_history.clear()
        for _ in range(cfg.RNN_SEQUENCE_LENGTH + 20):
            self.leader_q_history.append(self.initial_qpos.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))
            self.remote_q_history.append(self.initial_qpos.copy())
            self.remote_qd_history.append(np.zeros(self.n_joints))
        self.steps_remaining_in_warmup = self.warmup_steps
        self._last_gravity_torque = self._compute_gravity_compensation(self.initial_qpos)
        self._last_id_torque = np.zeros(self.n_joints)
        self._prev_id_torque = np.zeros(self.n_joints)
        self._last_teacher_action = np.zeros(self.n_joints)
        return self._get_observation(), {}
    def _precompute_trajectory(self, start_q):
        rollout_steps = self.max_episode_steps + cfg.ESTIMATOR_PREDICTION_HORIZON
        backup_state = (self.leader._q_current.copy(), self.leader._q_previous.copy(), self.leader._trajectory_time, self.leader._tick)
        temp_q = [start_q.copy()]
        temp_qd = [np.zeros(self.n_joints)]
        for _ in range(rollout_steps):
            q, qd, _, _, _, _ = self.leader.step()
            temp_q.append(q.copy())
            temp_qd.append(qd.copy())
        self._precomputed_trajectory_q = np.array(temp_q)
        self._precomputed_trajectory_qd = np.array(temp_qd)
        self.leader._q_current = backup_state[0]
        self.leader._q_previous = backup_state[1]
        self.leader._trajectory_time = backup_state[2]
        self.leader._tick = backup_state[3]
    def step(self, action):
        self.current_step += 1
        new_leader_q, new_leader_qd, _, _, _, _ = self.leader.step()
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        gravity_torque = self._compute_gravity_compensation(remote_q)
        self._last_gravity_torque = gravity_torque
        true_target_full = self.get_true_current_target()
        target_q_gt = true_target_full[:self.n_joints]
        target_qd_gt = true_target_full[self.n_joints:]
        denormalized_lstm_pred = None
        if self._current_predicted_state is not None:
            pred_norm_q = self._current_predicted_state[:self.n_joints]
            denormalized_lstm_pred = self._denormalize_q(pred_norm_q)
            self._cached_prediction_error = np.linalg.norm(denormalized_lstm_pred - target_q_gt)
        if self._use_ground_truth:
            teacher_target_q = target_q_gt
            teacher_target_qd = target_qd_gt
        else:
            if denormalized_lstm_pred is not None:
                teacher_target_q = denormalized_lstm_pred
                teacher_target_qd = target_qd_gt 
            else:
                teacher_target_q = target_q_gt
                teacher_target_qd = target_qd_gt
        teacher_action = self._compute_teacher_action(remote_q, remote_qd, teacher_target_q, teacher_target_qd)
        self._last_teacher_action = teacher_action.copy()
        self._last_id_torque = teacher_action.copy()
        if self.steps_remaining_in_warmup > 0:
            self.steps_remaining_in_warmup -= 1
            tau_total = gravity_torque + teacher_action
            actual_action = teacher_action 
        elif self._use_teacher:
            tau_total = gravity_torque + teacher_action
            actual_action = teacher_action
        else:
            tau_total = gravity_torque + action
            actual_action = action
        tau_total = np.clip(tau_total, -self.torque_limits, self.torque_limits)
        log_pred_q = denormalized_lstm_pred if denormalized_lstm_pred is not None else target_q_gt
        self.remote_robot.step(target_q=target_q_gt, target_qd=target_qd_gt, torque_input=tau_total, true_local_q=target_q_gt, predicted_q=log_pred_q)
        
        # [FIX] Scaled Reward by 10.0
        reward = 10.0 * np.exp(-5.0 * np.linalg.norm(target_q_gt - remote_q))
        
        remote_q_new, _ = self.remote_robot.get_joint_state()
        joint_error = np.linalg.norm(target_q_gt - remote_q_new)
        hist_len = len(self.leader_q_history)
        self._cached_delay_steps = 0 if self.current_step < self.grace_period_steps else self.delay_simulator.get_state_delay_steps(hist_len)
        terminated, term_penalty = self._check_termination(joint_error, remote_q_new)
        reward += term_penalty
        truncated = self.current_step >= self.max_episode_steps
        if self.render_mode == "human": self.render()
        info = self._get_info()
        info["actual_action"] = actual_action
        info["teacher_action"] = teacher_action
        info["is_teacher_mode"] = self._use_teacher
        info["is_ground_truth_mode"] = self._use_ground_truth
        info["true_state"] = true_target_full
        return self._get_observation(), reward, terminated, truncated, info

    def get_delayed_target_buffer(self, buffer_length):
        history_len = len(self.leader_q_history)
        delay_steps = self.delay_simulator.get_state_delay_steps(history_len)
        normalized_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        most_recent_idx = -1 - delay_steps
        buffer_seq = []
        for i in range(-buffer_length + 1, 1): 
            idx = np.clip(most_recent_idx + i, -history_len, -1)
            step_vector = self._normalize_input(self.leader_q_history[idx], self.leader_qd_history[idx], normalized_delay)
            buffer_seq.append(step_vector)
        return np.array(buffer_seq).flatten().astype(np.float32)
    def _get_observation(self):
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        self.remote_q_history.append(remote_q.copy())
        self.remote_qd_history.append(remote_qd.copy())
        remote_state_normalized = self._normalize_state(remote_q, remote_qd)
        robot_history_seq = []
        hist_len = len(self.remote_q_history)
        for i in range(-cfg.RNN_SEQUENCE_LENGTH, 0):
            idx = max(-hist_len, i)
            q_norm = (self.remote_q_history[idx] - cfg.Q_MEAN) / cfg.Q_STD
            qd_norm = (self.remote_qd_history[idx] - cfg.QD_MEAN) / cfg.QD_STD
            robot_history_seq.extend(q_norm)
            robot_history_seq.extend(qd_norm)
        robot_history_flat = np.array(robot_history_seq, dtype=np.float32)
        target_history_flat = self.get_delayed_target_buffer(cfg.RNN_SEQUENCE_LENGTH)
        return np.concatenate([remote_state_normalized, robot_history_flat, target_history_flat]).astype(np.float32)
    def get_true_current_target(self):
        if not self.leader_q_history: return np.concatenate([self.initial_qpos, np.zeros(self.n_joints)])
        return np.concatenate([self.leader_q_history[-1], self.leader_qd_history[-1]])
    def _check_termination(self, joint_error, remote_q):
        if not np.all(np.isfinite(remote_q)): return True, -100.0
        at_limits = (np.any(remote_q <= self.joint_limits_lower + cfg.JOINT_LIMIT_MARGIN) or np.any(remote_q >= self.joint_limits_upper - cfg.JOINT_LIMIT_MARGIN))
        high_error = joint_error > self.max_joint_error
        terminated = at_limits or high_error
        penalty = -10.0 if terminated else 0.0
        return terminated, penalty
    def _get_info(self):
        is_warmup = self.current_step < self.grace_period_steps
        true_state = self.get_true_current_target()
        return {"prediction_error": self._cached_prediction_error, "current_delay_steps": self._cached_delay_steps, "is_in_warmup": is_warmup, "true_state": true_state, "gravity_torque": self._last_gravity_torque.copy(), "teacher_action": self._last_teacher_action.copy()}
    def render(self): pass
    def close(self): pass