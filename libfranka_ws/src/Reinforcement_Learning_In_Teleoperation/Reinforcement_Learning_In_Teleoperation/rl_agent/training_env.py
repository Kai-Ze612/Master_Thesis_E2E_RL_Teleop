"""
Gymnasium Training environment.

The environment is 

Integrated with:
- DelaySimulator for setting network delay patterns.
- LocalRobotSimulator: create target joint positions and velocities.
- RemoteRobotSimulator: subscribe to RL output predicited targets and velocities, apply inverse dynamics and torque compensation.
"""

# RL library imports
import gymnasium as gym
from gymnasium import spaces  # in order to define the action and observation spaces

# Python standard libraries
import numpy as np
from collections import deque  # better for list function with fixed max length and faster appends/pops
from typing import Tuple, Dict, Any, Optional
import warnings
import matplotlib.pyplot as plt

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
    REWARD_PREDICTION_WEIGHT,
    REWARD_TRACKING_WEIGHT,
    REWARD_ERROR_SCALE,
    REWARD_VEL_PREDICTION_WEIGHT_FACTOR,
    RNN_SEQUENCE_LENGTH,
)


class TeleoperationEnvWithDelay(gym.Env):

    metadata = {'render_modes': ["human", "rgb_array"], 'render_fps': DEFAULT_CONTROL_FREQ}
    
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.LOW_DELAY,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_trajectory: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        # Initialize trajectory as simplest for simplicity
    ):
        super().__init__()

        # Render mode
        self.render_mode = render_mode
        self.viewer = None
        self.ax = None
        
        # History buffers for plotting
        self.plot_history_len = 1000
        self.hist_tracking_reward = deque(maxlen=self.plot_history_len)
        self.hist_prediction_reward = deque(maxlen=self.plot_history_len)
        self.hist_total_reward = deque(maxlen=self.plot_history_len)
        self._step_counter = 0
        
        # Environment parameters
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.control_freq = DEFAULT_CONTROL_FREQ
        self.current_step = 0
        self.episode_count = 0

        # Safety limits
        self.max_joint_error = MAX_JOINT_ERROR_TERMINATION  # RL termination threshold
        self.joint_limit_margin = JOINT_LIMIT_MARGIN 

        # Franka Panda robot parameters
        self.n_joints = N_JOINTS
        self.initial_qpos = INITIAL_JOINT_CONFIG.copy()
        self.joint_limits_lower = JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = JOINT_LIMITS_UPPER.copy()
        self.torque_limits = TORQUE_LIMITS.copy()
        
        # Initialize delay simulator
        self.delay_config = delay_config
        self.delay_simulator = DelaySimulator(
            control_freq=self.control_freq,
            config=delay_config,
            seed=seed
        )

        # Initialize Leader and Remote Robot
        self.trajectory_type = trajectory_type
        self.randomize_trajectory = randomize_trajectory
        self.leader = LocalRobotSimulator(
            trajectory_type=self.trajectory_type,
            randomize_params=self.randomize_trajectory
        )
        self.remote_robot = RemoteRobotSimulator()
    
        # History buffers for observations and actions
        self.action_history_len = ACTION_HISTORY_LEN
        self.target_history_len = TARGET_HISTORY_LEN

        # Calculate buffer sizes based on maximum possible delays
        max_obs_delay = self.delay_simulator._obs_delay_max_steps
        max_action_delay = self.delay_simulator._action_delay_steps
        leader_q_buffer_size = max(100, max_obs_delay + RNN_SEQUENCE_LENGTH + 20) # make sure the buffer has at least 100 entries and can fit the RNN sequence, 20 is the safety margin
        leader_qd_buffer_size = max(100, max_obs_delay + RNN_SEQUENCE_LENGTH + 20)
        action_buffer_size = max(100, max_action_delay + self.action_history_len + 20)

        self.leader_q_history = deque(maxlen=leader_q_buffer_size)
        self.leader_qd_history = deque(maxlen=leader_qd_buffer_size)
        self.action_history = deque(maxlen=action_buffer_size)
        
        # Store predicted target from policy (set via set_predicted_target())
        self._last_predicted_target: Optional[np.ndarray] = None
        
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
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state for every new episode."""
        
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_count += 1
        self._last_predicted_target = None  # Clear predicted target

        # Reset leader and remote robot
        leader_start_q, _ = self.leader.reset(seed=seed, options=options)
        self.remote_robot.reset(initial_qpos=self.initial_qpos)
        
        # Clear history buffers
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        self.action_history.clear()

        # Pre-fill leader history buffer
        max_history_needed = self.delay_simulator._obs_delay_max_steps + self.target_history_len + 5
        for _ in range(max_history_needed):
            self.leader_q_history.append(leader_start_q.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))
            
        # Pre-fill action history buffer
        action_buffer_len = self.delay_simulator._action_delay_steps + self.action_history_len + 5
        for _ in range(action_buffer_len):
            self.action_history.append(np.zeros(self.n_joints))
        
        return self._get_observation(), self._get_info()

    def set_predicted_target(self, predicted_target: np.ndarray) -> None:
        """
        - predict the current local robot position and velocity based on the delayed observations.
        - This method must be called before env.step() to set the predicted target from the policy.
        - Output: 14 D array: [predicted_q (7,), predicted_qd (7,)]
        """
        if predicted_target.shape[0] != N_JOINTS * 2:
            raise ValueError(f"predicted_target must have shape ({N_JOINTS * 2},), got {predicted_target.shape}")
        
        self._last_predicted_target = predicted_target.copy()

    def _get_delayed_q(self) -> np.ndarray:
        """Get the delayed target joint position from the history buffer."""
        buffer_len = len(self.leader_q_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(buffer_len)
        delay_index = min(delay_steps, buffer_len - 1)
        return self.leader_q_history[-delay_index - 1].copy()
    
    def _get_delayed_qd(self) -> np.ndarray:
        """Get the delayed target joint velocity from the history buffer."""
        buffer_len = len(self.leader_qd_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(buffer_len)
        delay_index = min(delay_steps, buffer_len - 1)
        return self.leader_qd_history[-delay_index - 1].copy()

    def _get_delayed_action(self) -> np.ndarray:
        """Get the delayed action from the history buffer."""
        buffer_len = len(self.action_history)
        delay_steps = self.delay_simulator.get_action_delay_steps()
        delay_index = min(delay_steps, buffer_len - 1)
        return self.action_history[-delay_index - 1].copy()

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep in the environment given the predited target and RL action (tau compensation).
    
        Returns: observation, reward, terminated, truncated, info
        """
        
        self.current_step += 1
        self._step_counter += 1
        
        # Leader generate trajectory point
        new_leader_q, new_leader_qd, _, _, _, _ = self.leader.step()
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())

        # Store action in history buffer
        self.action_history.append(action.copy())

        # Get delay input for remote robot
        delayed_q = self._get_delayed_q()
        delayed_action = self._get_delayed_action()

        # Determine target for remote robot based on last predicted target
        if self._last_predicted_target is not None:
            # The remote robot targets the position component of the agent's prediction (the q value)
            target_q_for_remote = self._last_predicted_target[:N_JOINTS]
            
            # The torque compensation is the immediate RL action (input 'action')
            torque_compensation_for_remote = action
        else:
            # Fallback for the very first step before the agent predicts (should not happen in VecEnv)
            target_q_for_remote = self.initial_qpos.copy() # Use a safe, static target
            torque_compensation_for_remote = np.zeros(N_JOINTS)
        
        # Apply target and torque compensation to remote robot
        step_info = self.remote_robot.step(
            target_q=target_q_for_remote,      
            torque_compensation=torque_compensation_for_remote,
        )
        
        # Get remote robot current state after implementing action
        remote_q, remote_qd = self.remote_robot.get_joint_state()

        # Calculate reward (includes prediction + tracking)
        reward = self._calculate_reward(action)

        # Update history buffers for plotting
        self.hist_total_reward.append(reward)
        
        # Calculate individual tracking/prediction rewards to store them
        r_prediction_weighted = REWARD_PREDICTION_WEIGHT * self._calculate_reward_components()['r_prediction']
        r_tracking_weighted = REWARD_TRACKING_WEIGHT * self._calculate_reward_components()['r_tracking']

        self.hist_prediction_reward.append(r_prediction_weighted)
        self.hist_tracking_reward.append(r_tracking_weighted)
        
        # Check termination
        if self._last_predicted_target is not None:
            predicted_q = self._last_predicted_target[:N_JOINTS]
            joint_error = np.linalg.norm(predicted_q - remote_q)
        else:
            # Fallback: use error from initial position
            joint_error = np.linalg.norm(self.initial_qpos - remote_q)
            
        terminated, term_penalty = self._check_termination(joint_error, remote_q)
        if terminated:
            reward += term_penalty

        # Check truncation
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()        
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector.
        
        Observation components:
            - remote_q: Current remote robot joint positions (7,)
            - remote_qd: Current remote robot joint velocities (7,)
            - predicted_q: Last predicted target joint positions (7,)
            - predicted_qd: Last predicted target joint velocities (7,)
            - target_q_history: History of target joint positions (N_JOINTS * TARGET_HISTORY_LEN,)
            - target_qd_history: History of target joint velocities (N_JOINTS * TARGET_HISTORY_LEN,)
            - delay_magnitude: Current observation delay in timesteps (1,)
            - action_history: History of past actions (N_JOINTS * ACTION_HISTORY_LEN,)
        """
        
        # Get current remote state
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        
        # Get predicted target from last set_predicted_target() call
        if self._last_predicted_target is not None:
            predicted_q = self._last_predicted_target[:N_JOINTS] # Get only the 7 pos values
            predicted_qd = self._last_predicted_target[N_JOINTS:] # Get only the 7 vel values
        else:
            # Handle case before first prediction (e.g., at reset)
            predicted_q = np.zeros(N_JOINTS) # Must return a 7-dim vector
            predicted_qd = np.zeros(N_JOINTS) # Must return a 7-dim vector

        # Store target history
        target_q_history = []
        target_qd_history = []
        for i in range(1, self.target_history_len + 1):
            idx = -i - 1 if i < len(self.leader_q_history) else 0
            target_q_history.append(self.leader_q_history[idx].copy())
            target_qd_history.append(self.leader_qd_history[idx].copy())

        target_q_history = np.array(target_q_history).flatten()
        target_qd_history = np.array(target_qd_history).flatten()

        # Store action history
        action_history = []
        for i in range(1, self.action_history_len + 1):
            idx = -i - 1 if i < len(self.action_history) else 0
            action_history.append(self.action_history[idx].copy())
        action_history = np.array(action_history).flatten()

        # Pad histories if needed (for initial steps)
        expected_q_len = N_JOINTS * self.target_history_len
        expected_qd_len = N_JOINTS * self.target_history_len
        expected_act_len = N_JOINTS * self.action_history_len

        target_q_history_padded = np.pad(
            target_q_history, 
            (max(0, expected_q_len - len(target_q_history)), 0), 
            mode='edge'
        )
        target_qd_history_padded = np.pad(
            target_qd_history, 
            (max(0, expected_qd_len - len(target_qd_history)), 0), 
            mode='edge'
        )
        action_history_padded = np.pad(
            action_history, 
            (max(0, expected_act_len - len(action_history)), 0), 
            mode='edge'
        )

        # Store delay magnitude
        obs_delay_steps = self.delay_simulator.get_observation_delay_steps(len(self.leader_q_history))
        delay_magnitude = np.array([obs_delay_steps / 100.0])

        # Concatenate all components
        obs_vec = np.concatenate([
            remote_q,
            remote_qd,
            predicted_q,
            predicted_qd,
            target_q_history_padded,
            target_qd_history_padded,
            delay_magnitude,
            action_history_padded
        ]).astype(np.float32)
        
        # Validate observation shape
        if obs_vec.shape[0] != self.observation_space.shape[0]:
            warnings.warn(f"Observation dimension mismatch: got {obs_vec.shape[0]}, expected {self.observation_space.shape[0]}")

        return obs_vec

    def get_true_current_target(self) -> np.ndarray:
        """Get ground truth current target for RNN training."""
        
        if not self.leader_q_history or not self.leader_qd_history:
            # Return initial state if history is empty
            return np.concatenate([self.initial_qpos, np.zeros(self.n_joints)]).astype(np.float32)
        
        current_q_gt = self.leader_q_history[-1].copy()
        current_qd_gt = self.leader_qd_history[-1].copy()
        return np.concatenate([current_q_gt, current_qd_gt]).astype(np.float32)

    def get_delayed_target_buffer(self, buffer_length: int) -> np.ndarray:
        """Get delayed target sequence for state predictor (LSTM) input."""
        
        history_len = len(self.leader_q_history)
        
        # Handle empty history
        if history_len == 0:
            initial_state = np.concatenate([self.initial_qpos, np.zeros(self.n_joints)])
            return np.tile(initial_state, buffer_length).astype(np.float32)

        # Get current observation delay
        obs_delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
        
        # Most recent delayed observation index
        most_recent_delayed_idx = -(obs_delay_steps + 1)
        
        # Oldest index we need
        oldest_idx = most_recent_delayed_idx - buffer_length + 1
        
        buffer_q = []
        buffer_qd = []
        
        # Iterate from oldest to most recent (FORWARD in time)
        for i in range(oldest_idx, most_recent_delayed_idx + 1):
            # Clip to valid range [-history_len, -1]
            safe_idx = np.clip(i, -history_len, -1)
            buffer_q.append(self.leader_q_history[safe_idx].copy())
            buffer_qd.append(self.leader_qd_history[safe_idx].copy())
        
        # Flatten the for input data
        buffer = np.stack([np.concatenate([q, qd]) for q, qd in zip(buffer_q, buffer_qd)]).flatten()
        
        # Validate shape
        expected_shape = buffer_length * N_JOINTS * 2
        if buffer.shape[0] != expected_shape:
            warnings.warn(f"Delayed buffer shape mismatch: got {buffer.shape[0]}, expected {expected_shape}")
        
        return buffer.astype(np.float32)
        
    def get_remote_state(self) -> np.ndarray:
        """Get current remote robot state (real-time, no delay)."""
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        return np.concatenate([remote_q, remote_qd]).astype(np.float32)
    
    def get_current_observation_delay(self) -> int:
        """Get the current observation delay in timesteps."""

        history_len = len(self.leader_q_history)
        return self.delay_simulator.get_observation_delay_steps(history_len)
    
    def _get_adaptive_position_scale(self, error: float) -> float:
        """Adaptive scaling: gentle for large errors, steep for small"""
        error_mm = error * 1000
        
        if error_mm > 200:
            return 10.0
        elif error_mm > 100:
            alpha = (200 - error_mm) / 100.0
            return 10.0 + 25.0 * alpha
        else:
            return 35.0

    def _calculate_reward_components(self) -> Dict[str, float]:
        r_prediction = 0.0
        r_tracking = 0.0
        
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        
        if self._last_predicted_target is not None:
            true_target = self.get_true_current_target()
            true_target_q = true_target[:N_JOINTS]
            true_target_qd = true_target[N_JOINTS:]
            predicted_q = self._last_predicted_target[:N_JOINTS]
            predicted_qd = self._last_predicted_target[N_JOINTS:]

            # Adaptive scaling
            pos_pred_error = np.linalg.norm(predicted_q - true_target_q)
            pos_scale = self._get_adaptive_position_scale(pos_pred_error)
            
            tracking_pos_error = np.linalg.norm(predicted_q - remote_q)
            track_scale = self._get_adaptive_position_scale(tracking_pos_error)
            
            # Position rewards
            r_pos_prediction = np.exp(-pos_scale * pos_pred_error**2)
            r_pos_tracking = np.exp(-track_scale * tracking_pos_error**2)
            
            # Velocity rewards - CRITICAL FIX: 5× STRONGER
            vel_pred_error = np.linalg.norm(predicted_qd - true_target_qd)
            r_vel_prediction = np.exp(-pos_scale * 5.0 * vel_pred_error**2)  # Changed from 1.0 to 5.0
            
            tracking_vel_error = np.linalg.norm(predicted_qd - remote_qd)
            r_vel_tracking = np.exp(-track_scale * 5.0 * tracking_vel_error**2)  # Changed from 1.0 to 5.0
            
            # Normalized arithmetic mean
            r_prediction = (r_pos_prediction + REWARD_VEL_PREDICTION_WEIGHT_FACTOR * r_vel_prediction) / \
                        (1.0 + REWARD_VEL_PREDICTION_WEIGHT_FACTOR)
            r_tracking = (r_pos_tracking + REWARD_VEL_PREDICTION_WEIGHT_FACTOR * r_vel_tracking) / \
                        (1.0 + REWARD_VEL_PREDICTION_WEIGHT_FACTOR)
            
            return {
                'r_prediction': r_prediction,
                'r_tracking': r_tracking
            }
        
    def _calculate_reward(
        self,
        action: np.ndarray,  # tau_compensation
    ) -> float:
        """
        Calculate dense reward combining prediction and tracking accuracy.
        """
        
        components = self._calculate_reward_components()
        total_reward = components['r_tracking']
        
        # Add small penalty for large actions to encourage smoother control
        action_penalty = 0.01 * np.sum(action**2)
        total_reward -= action_penalty
        
        # Logging (every 1000 steps)
        if self.current_step % 1000 == 0:
            r_prediction = components['r_prediction']
            r_tracking = components['r_tracking']
            
            if self._last_predicted_target is not None:
                true_target = self.get_true_current_target()
                predicted_q = self._last_predicted_target[:N_JOINTS]
                predicted_qd = self._last_predicted_target[N_JOINTS:]
                true_target_q = true_target[:N_JOINTS]
                true_target_qd = true_target[N_JOINTS:]
                remote_q, remote_qd = self.remote_robot.get_joint_state()
                
                pred_pos_error = np.linalg.norm(predicted_q - true_target_q)
                pred_vel_error = np.linalg.norm(predicted_qd - true_target_qd)
                track_pos_error = np.linalg.norm(predicted_q - remote_q)
                track_vel_error = np.linalg.norm(predicted_qd - remote_qd)
                
                print(f"\n[Reward Debug - Step {self.current_step}]")
                print(f"  RNN Prediction Accuracy (NOT part of RL reward):")
                print(f"    Pos Error: {pred_pos_error*1000:.1f}mm → Component: {r_prediction:.3f}")
                print(f"    Vel Error: {pred_vel_error:.3f} rad/s")
                print(f"  RL Tracking Performance (to predicted goal):")
                print(f"    Pos Error: {track_pos_error*1000:.1f}mm → Component: {r_tracking:.3f}")
                print(f"    Vel Error: {track_vel_error:.3f} rad/s")
                print(f"    Weighted Tracking: {REWARD_TRACKING_WEIGHT * r_tracking:.3f}")
                print(f"  TOTAL RL REWARD: {total_reward:.3f}")

        return total_reward
    
    def _check_termination(
        self,
        joint_error: float,
        remote_q: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Termination occurs if:
            1. Joint limits are approached (within margin)
            2. Joint error is too high (> max_joint_error)
            3. Joint error is NaN (numerical instability)
        """
        
        # Check joint limits
        at_limits = (
            np.any(remote_q <= self.joint_limits_lower + self.joint_limit_margin) or
            np.any(remote_q >= self.joint_limits_upper - self.joint_limit_margin)
        )
        
        # Check for instability or excessive error
        high_error = np.isnan(joint_error) or joint_error > self.max_joint_error

        terminated = at_limits or high_error
        penalty = -10.0 if terminated else 0.0
        
        return terminated, penalty

    def _get_info(self) -> Dict[str, Any]:
        """
        info for debugging and analysis.
        
        components:
            - real_time_joint_error: ||true_q[t] - remote_q[t]||
            - prediction_error: ||predicted_q[t] - true_q[t]||
            - current_delay_steps: current observation delay in timesteps
        """
        
        info_dict = {}
        remote_q, _ = self.remote_robot.get_joint_state()

        # Calculate real-time tracking error
        if self.leader_q_history:
            true_target_q = self.leader_q_history[-1]
            real_time_pos_error_norm = np.linalg.norm(true_target_q - remote_q)
            info_dict['real_time_joint_error'] = real_time_pos_error_norm

            # Calculate prediction error if available
            if self._last_predicted_target is not None:
                predicted_q = self._last_predicted_target[:N_JOINTS]
                info_dict['prediction_error'] = np.linalg.norm(predicted_q - true_target_q)
            else:
                info_dict['prediction_error'] = np.nan
        else:
            # Initial state
            info_dict['real_time_joint_error'] = 0.0
            info_dict['prediction_error'] = np.nan

        info_dict['current_delay_steps'] = self.get_current_observation_delay()

        return info_dict

    def render(self) -> None:
        """Render the environment for human viewing (live plot)."""
        if self.render_mode != "human":
            return

        if self.viewer is None:
            # Initialize a 3-subplot figure for key performance metrics
            self.viewer, self.ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            self.viewer.suptitle(f'Live Teleoperation Tracking - Run {self.episode_count}', fontsize=12)
            
            # Subplot 1: Total Reward
            self.line1, = self.ax[0].plot([], [], label='TOTAL Step Reward', color='green')
            self.ax[0].set_ylabel('Total Reward')
            self.ax[0].legend(loc='upper right')
            
            # Subplot 2: Tracking Component
            self.line2, = self.ax[1].plot([], [], label='Tracking Reward (Weighted)', color='blue')
            self.ax[1].set_ylabel('Tracking Reward')
            self.ax[1].legend(loc='upper right')
            
            # Subplot 3: Prediction Component
            self.line3, = self.ax[2].plot([], [], label='Prediction Reward (Weighted)', color='red')
            self.ax[2].set_ylabel('Prediction Reward')
            self.ax[2].set_xlabel(f'Time Steps (History Length: {self.plot_history_len})')
            self.ax[2].legend(loc='upper right')
            
            plt.ion() # Turn on interactive mode for non-blocking plot updates
            plt.show(block=False)

        # 1. Update Plot Title and X-axis Data
        x_data = np.arange(self._step_counter - len(self.hist_total_reward) + 1, 
                           self._step_counter + 1)
        
        # 2. Update Y-axis Data
        self.line1.set_data(x_data, self.hist_total_reward)
        self.line2.set_data(x_data, self.hist_tracking_reward)
        self.line3.set_data(x_data, self.hist_prediction_reward)

        # 3. Autoscale and Redraw (Crucial for live updating)
        for ax in self.ax:
            ax.relim()      # Recalculate limits based on new data
            ax.autoscale_view() # Rescale axes
        
        self.viewer.canvas.draw_idle()
        self.viewer.canvas.flush_events()

    def close(self) -> None:
        """Clean up resources and close the Matplotlib figure."""
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None