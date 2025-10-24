"""
Gymnasium environment for teleoperation with variable network delay.

Architecture:
- Leader (LocalRobotSimulator): Generates reference trajectories
- Delay Simulator: Models network observation and action delays
- Follower (RemoteRobotSimulator): Tracks leader with delay-adaptive control
- RL Agent: Learns torque corrections on top of baseline controller

"""

# RL library imports
import gymnasium as gym
from gymnasium import spaces

# Python standard libraries
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional
import warnings

# Custom imports
from .local_robot_simulator import LocalRobotSimulator, TrajectoryType
from .remote_robot_simulator import RemoteRobotSimulator
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import (
    DelaySimulator,
    ExperimentConfig
)

# Configuration imports
from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    INITIAL_JOINT_CONFIG,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    MAX_EPISODE_STEPS,
    MAX_JOINT_ERROR_TERMINATION,
    ACTION_HISTORY_LEN,
    TARGET_HISTORY_LEN,
    DEFAULT_CONTROL_FREQ,
    JOINT_LIMIT_MARGIN,
    N_JOINTS,
    TORQUE_LIMITS,
    MAX_TORQUE_COMPENSATION,
    OBS_DIM,
    STATE_BUFFER_LENGTH,
    REWARD_PREDICTION_WEIGHT,
    REWARD_TRACKING_WEIGHT,
    REWARD_ERROR_SCALE,
    REWARD_VEL_PREDICTION_WEIGHT_FACTOR,
)

class TeleoperationEnvWithDelay(gym.Env):

    metadata = {'render_modes': []}

    def __init__(
        self,
        experiment_config: ExperimentConfig = ExperimentConfig.MEDIUM_DELAY,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_trajectory: bool = False,
        **kwargs
    ):
        """ Initialize the joint space teleoperation environment with delay simulation. """

        self.max_episode_steps = MAX_EPISODE_STEPS
        self.control_freq = DEFAULT_CONTROL_FREQ
        self.current_step = 0
        self.episode_count = 0

        # Safety limits
        self.max_joint_error = MAX_JOINT_ERROR_TERMINATION
        self.joint_limit_margin = JOINT_LIMIT_MARGIN

        # Franka Panda robot parameters
        self.n_joints = N_JOINTS
        self.initial_qpos = INITIAL_JOINT_CONFIG.copy()
        self.joint_limits_lower = JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = JOINT_LIMITS_UPPER.copy()
        self.torque_limits = TORQUE_LIMITS.copy()
        
        # Initialize delay simulator
        self.experiment_config = experiment_config
        self.delay_simulator = DelaySimulator(
            control_freq=self.control_freq,
            config=experiment_config,
            seed=None
        )
        
        # Initialize Leader and Remote Robot
        self.trajectory_type = trajectory_type
        self.randomize_trajectory = randomize_trajectory
        
        self.leader = LocalRobotSimulator(
            trajectory_type=self.trajectory_type,
            randomize_params=self.randomize_trajectory
        )
        self.remote_robot = RemoteRobotSimulator()
    
        # History buffer
        self.action_history_len = ACTION_HISTORY_LEN
        self.target_history_len = TARGET_HISTORY_LEN

        max_obs_delay = self.delay_simulator._obs_delay_max_steps
        max_action_delay = self.delay_simulator._action_delay_steps
        leader_q_buffer_size = max(100, max_obs_delay + self.target_history_len + 5)
        action_buffer_size = max(50, max_action_delay + self.action_history_len + 5)

        self.leader_q_history = deque(maxlen=leader_q_buffer_size)
        self.leader_qd_history = deque(maxlen=leader_q_buffer_size)
        self.action_history = deque(maxlen=action_buffer_size)
        
        # Action and observation spaces setup
        self.action_space = spaces.Box(
            low=-MAX_TORQUE_COMPENSATION.copy(),
            high=MAX_TORQUE_COMPENSATION.copy(),
            shape=(self.n_joints,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        
        self._last_predicted_target: Optional[np.ndarray] = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to the initial state for a new episode."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_count += 1
        self._last_predicted_target = None
        
        # Reset leader and remote robot
        leader_start_q, _ = self.leader.reset(seed=seed, options=options)
        self.remote_robot.reset(initial_qpos=self.initial_qpos)

        # Clear history buffers
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        self.action_history.clear()

        max_history_needed = self.delay_simulator._obs_delay_max_steps + self.target_history_len + 5
        for _ in range(max_history_needed):
            self.leader_q_history.append(leader_start_q.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))

        for _ in range(self.delay_simulator._action_delay_steps + self.action_history_len + 5):
            self.action_history.append(np.zeros(self.n_joints))

        return self._get_observation(), self._get_info()

    def _get_delayed_q(self) -> np.ndarray:
        """Gets the delayed target joint position from the history buffer."""
        buffer_len = len(self.leader_q_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(buffer_len)
        delay_index = min(delay_steps, buffer_len - 1)
        return self.leader_q_history[-delay_index - 1].copy()

    def _get_delayed_action(self) -> np.ndarray:
        """Gets the delayed action from the history buffer."""
        buffer_len = len(self.action_history)
        delay_steps = self.delay_simulator.get_action_delay_steps()
        delay_index = min(delay_steps, buffer_len - 1)
        return self.action_history[-delay_index - 1].copy()

    def _set_predicted_target(self, predicted_target: np.ndarray) -> None:
        """ Gets the predicted local robot state"""
        self._last_predicted_target = predicted_target.copy()
        
    def step(
        self,
        action: np.ndarray # This is tau compensation from policy
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step in the environment."""
        self.current_step += 1

        # Advanced Leader
        new_leader_q, _, _, _, leader_info = self.leader.step()
        new_leader_qd = leader_info.get('joint_vel', np.zeros(self.n_joints))
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())

        # Store RL action in history
        self.action_history.append(action.copy())
        
        # Get delayed inputs for the remote robot
        delayed_q_target_for_pd = self._get_delayed_q()
        delayed_rl_action = self._get_delayed_action()
        
        # Step remote robot simulator
        self.remote_robot.step(
            target_q=delayed_q_target_for_pd,    # Baseline PD uses delayed target
            torque_compensation=delayed_rl_action # Add RL's (potentially delayed) action
        )
        
        # Calculate real-time tracking error for reward
        remote_q, _ = self.remote_robot.get_joint_state()
        # 'new_leader_q' is the true target at the current time t
        real_time_pos_error_norm = np.linalg.norm(new_leader_q - remote_q)
        
        # Calculate reward using the new dense function
        reward = self._calculate_reward(real_time_pos_error_norm, action)
        
        # Check termination and truncation
        terminated, term_penalty = self._check_termination(real_time_pos_error_norm, remote_q)
        reward += term_penalty
        truncated = self.current_step >= self.max_episode_steps

        # Get next observation
        observation = self._get_observation()
        info = self._get_info() # Get info after calculating errors

        # Clear the stored prediction after using it in reward calculation
        # Ensures old predictions aren't used if set_predicted_target isn't called next step
        self._last_predicted_target = None

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Assembles the observation vector for the RL agent."""
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        delayed_target_q = self._get_delayed_q()

        # Getting History Buffers
        # Ensure history buffers are accessed safely, even if short
        q_hist = list(self.leader_q_history)
        qd_hist = list(self.leader_qd_history)
        act_hist = list(self.action_history)
        
        target_q_history = np.array(q_hist[-TARGET_HISTORY_LEN:]).flatten()
        target_qd_history = np.array(qd_hist[-TARGET_HISTORY_LEN:]).flatten()
        action_history = np.array(act_hist[-ACTION_HISTORY_LEN:]).flatten()
        
        # Pad histories if they are shorter than required
        expected_q_len = N_JOINTS * TARGET_HISTORY_LEN
        expected_qd_len = N_JOINTS * TARGET_HISTORY_LEN
        expected_act_len = N_JOINTS * ACTION_HISTORY_LEN
        
        target_q_history_padded = np.pad(target_q_history, (max(0, expected_q_len - len(target_q_history)), 0), mode='edge')
        target_qd_history_padded = np.pad(target_qd_history, (max(0, expected_qd_len - len(target_qd_history)), 0), mode='edge')
        action_history_padded = np.pad(action_history, (max(0, expected_act_len - len(action_history)), 0), mode='edge')

        obs_delay_steps = self.delay_simulator.get_observation_delay_steps(len(self.leader_q_history))
        delay_magnitude = np.array([obs_delay_steps / 100.0]) # Scaled
        
        obs_vec = np.concatenate([
            remote_q,
            remote_qd,
            delayed_target_q,
            target_q_history_padded,
            target_qd_history_padded,
            delay_magnitude,
            action_history_padded
        ]).astype(np.float32)
        
        # Use the shape defined in the observation space for validation
        if obs_vec.shape[0] != self.observation_space.shape[0]:
            print(f"--- Observation Dimension Warning ---")

        return obs_vec

    def get_true_current_target(self) -> np.ndarray:
        """ Get ground truth current target for supervised learning."""
        if not self.leader_q_history or not self.leader_qd_history:
            # Return initial state if history is empty
            return np.concatenate([self.initial_qpos, np.zeros(self.n_joints)]).astype(np.float32)
        current_q = self.leader_q_history[-1].copy()
        current_qd = self.leader_qd_history[-1].copy()
        return np.concatenate([current_q, current_qd]).astype(np.float32)
    
    def get_delayed_target_buffer(self, buffer_length: int = None) -> np.ndarray:
        """ Get delayed target for state predictor input."""
        
        history_len = len(self.leader_q_history)
        if history_len == 0:
             initial_state = np.concatenate([self.initial_qpos, np.zeros(self.n_joints)])
             return np.tile(initial_state, buffer_length).astype(np.float32)

        obs_delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)

        buffer_q = []
        buffer_qd = []
        start_idx = -(obs_delay_steps + 1)
        end_idx = start_idx - buffer_length

        for i in range(start_idx, end_idx, -1):
            # Clip index to valid range [-history_len, -1]
            safe_idx = np.clip(i, -history_len, -1)
            buffer_q.append(self.leader_q_history[safe_idx].copy())
            buffer_qd.append(self.leader_qd_history[safe_idx].copy())

        # The loop collects items from newest_delayed back to oldest_in_buffer.
        # We need chronological order [oldest ... newest_delayed] for LSTM.
        buffer_q.reverse()
        buffer_qd.reverse()

        # Flatten into [q0, qd0, q1, qd1, ...]
        buffer = np.stack([np.concatenate([q, qd]) for q, qd in zip(buffer_q, buffer_qd)]).flatten()

        # Ensure correct final shape
        expected_shape = buffer_length * N_JOINTS * 2
        if buffer.shape[0] != expected_shape:
            print(f"--- Delayed Target Buffer Dimension Warning ---")

        return buffer.astype(np.float32)
    
    def get_remote_state(self) -> np.ndarray:
        """Get current remote robot state (real-time, no delay)."""
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        return np.concatenate([remote_q, remote_qd]).astype(np.float32)
    
    def get_current_observation_delay(self) -> int:
        """ Get the current observation delay in timesteps."""
        history_len = len(self.leader_q_history)
        return self.delay_simulator.get_observation_delay_steps(history_len)
    
    def _calculate_reward(
        self,
        real_time_pos_error_norm: float, # ||true_q[t] - remote_q[t]||
        action: np.ndarray # tau_compensation
    ) -> float:
        """MODIFIED: Multi-objective dense reward."""

        # Objective 1: Prediction Accuracy (Dense Signal)
        r_prediction = 0.0
        pos_pred_error = np.nan # Initialize position error
        vel_pred_error = np.nan # Initialize velocity error

        if self._last_predicted_target is not None:
            # Get ground truth current target
            true_target = self.get_true_current_target()
            true_target_q = true_target[:N_JOINTS]
            true_target_qd = true_target[N_JOINTS:]

            # Get predicted target
            predicted_q = self._last_predicted_target[:N_JOINTS]
            predicted_qd = self._last_predicted_target[N_JOINTS:]

            # Position prediction error and reward
            pos_pred_error = np.linalg.norm(predicted_q - true_target_q)
            r_pos_prediction = np.exp(-REWARD_ERROR_SCALE * pos_pred_error**2)

            # Velocity prediction error and reward
            vel_pred_error = np.linalg.norm(predicted_qd - true_target_qd)
            r_vel_prediction = np.exp(-REWARD_ERROR_SCALE * 0.5 * vel_pred_error**2)

            # Combine the two prediction rewards
            r_prediction = r_pos_prediction + REWARD_VEL_PREDICTION_WEIGHT_FACTOR * r_vel_prediction
        else:
            r_prediction = 0.0 # Reward is 0

        # Objective 2: Tracking Accuracy
        r_pos_tracking = np.exp(-REWARD_ERROR_SCALE * real_time_pos_error_norm**2)

        # Velocity tracking reward (good for dynamics)
        if self.leader_qd_history: # Check if history exists
            true_target_qd = self.leader_qd_history[-1] # true qd[t]
            _, remote_qd = self.remote_robot.get_joint_state() # remote qd[t]
            vel_track_error = np.linalg.norm(true_target_qd - remote_qd)
            r_vel_tracking = np.exp(-REWARD_ERROR_SCALE * 0.5 * vel_track_error**2) # Scale differently?
        else:
            r_vel_tracking = 0.0 # No velocity history yet

        r_tracking = r_pos_tracking + 0.3 * r_vel_tracking # Combine pos and vel tracking

        # Weighted Combination
        total_reward = (
            REWARD_PREDICTION_WEIGHT * r_prediction +
            REWARD_TRACKING_WEIGHT * r_tracking
        )

        # Printout logging information
        if self.current_step % 500 == 0 and not np.isnan(pos_pred_error):
            print(f"\n[Reward Step {self.current_step}]")
            print(f"  Prediction:")
            print(f"    Pos Err: {pos_pred_error*1000:.1f}mm -> Rew: {r_pos_prediction:.3f}")
            print(f"    Vel Err: {vel_pred_error:.3f} -> Rew: {r_vel_prediction:.3f} (Factor: {REWARD_VEL_PREDICTION_WEIGHT_FACTOR})")
            print(f"    Combined Pred Rew: {r_prediction:.3f} (Weight: {REWARD_PREDICTION_WEIGHT})")
            print(f"  Tracking:")
            print(f"    Pos Err: {real_time_pos_error_norm*1000:.1f}mm -> Rew: {r_pos_tracking:.3f}")
            print(f"    Vel Err: {vel_track_error:.3f} -> Rew: {r_vel_tracking:.3f} (Factor: 0.3)")
            print(f"    Combined Track Rew: {r_tracking:.3f} (Weight: {REWARD_TRACKING_WEIGHT})")
            print(f"  TOTAL REWARD: {total_reward:.3f}")

        return total_reward
    
    def _check_termination(self, joint_error: float, remote_q: np.ndarray) -> Tuple[bool, float]:
        """Checks termination conditions."""
        at_limits = np.any(remote_q <= self.joint_limits_lower + self.joint_limit_margin) or \
                    np.any(remote_q >= self.joint_limits_upper - self.joint_limit_margin)
        # Check for NaN or excessively large errors which indicate instability
        high_error = np.isnan(joint_error) or joint_error > self.max_joint_error

        terminated = at_limits or high_error
        # Consistent penalty for termination
        return terminated, -10.0 if terminated else 0.0

    def _get_info(self) -> Dict[str, Any]:
        """Returns diagnostic information."""
        info_dict = {}
        remote_q, _ = self.remote_robot.get_joint_state()

        # Calculate real-time error internally
        if self.leader_q_history: # Check if history exists
            true_target_q = self.leader_q_history[-1]
            real_time_pos_error_norm = np.linalg.norm(true_target_q - remote_q)
            info_dict['real_time_joint_error'] = real_time_pos_error_norm

            # Calculate prediction error if available
            if self._last_predicted_target is not None:
                predicted_q = self._last_predicted_target[:N_JOINTS]
                info_dict['prediction_error'] = np.linalg.norm(predicted_q - true_target_q)
            else:
                info_dict['prediction_error'] = np.nan # Prediction wasn't available/used
        else: # Handle initial state
             info_dict['real_time_joint_error'] = 0.0
             info_dict['prediction_error'] = np.nan

        info_dict['current_delay_steps'] = self.get_current_observation_delay()
        # Add other info if needed

        return info_dict

    def render(self) -> None:
        """Rendering is not implemented."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass
