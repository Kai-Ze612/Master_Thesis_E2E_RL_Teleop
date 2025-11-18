"""
Gymnasium Training environment.

Pipeline:
1. LocalRobotSimulator: generates target trajectory (joint positions + velocities).
2. DelaySimulator: adding observation delays in receiving target from LocalRobotSimulator
3. LSTM State Estimator (pre-trained, frozen): receives the delay observation sequence and predicts the current target state.
4. RL Agent: based on the predicted target, outputs torque compensation action.
5. Apply PD control + torque compensation on RemoteRobotSimulator.
6. Adding action delays before applying to RemoteRobotSimulator.
7. Calculate reward based on true target from LocalRobotSimulator and current remote robot state.
"""

# RL library imports
import gymnasium as gym
from gymnasium import spaces  # in order to define the action and observation spaces

# Python standard libraries
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional
import warnings
import matplotlib.pyplot as plt

# Custom imports
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import LocalRobotSimulator, TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.remote_robot_simulator import RemoteRobotSimulator
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator

# Configuration imports
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    INITIAL_JOINT_CONFIG,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    MAX_EPISODE_STEPS,
    MAX_JOINT_ERROR_TERMINATION,
    DEFAULT_CONTROL_FREQ,
    JOINT_LIMIT_MARGIN,
    TORQUE_LIMITS,
    MAX_TORQUE_COMPENSATION,
    OBS_DIM,
    REMOTE_HISTORY_LEN,
    TRACKING_ERROR_SCALE,
    VELOCITY_ERROR_SCALE,
    ACTION_PENALTY_WEIGHT,
    RNN_SEQUENCE_LENGTH,
    LSTM_MODEL_PATH,
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
        pretrained_estimator_path: Optional[str] = None,
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

        # Action delay
        self.action_delay_steps = self.delay_simulator.get_action_delay_steps()
        self.torque_buffer = deque() if self.action_delay_steps > 0 else None
        
        # Initialize Leader and Remote Robot
        self.trajectory_type = trajectory_type
        self.randomize_trajectory = randomize_trajectory
        self.leader = LocalRobotSimulator(
            trajectory_type=self.trajectory_type,
            randomize_params=self.randomize_trajectory
        )
        self.remote_robot = RemoteRobotSimulator()

        # Calculate buffer sizes based on maximum possible delays
        max_obs_delay = self.delay_simulator._obs_delay_max_steps
        leader_q_buffer_size = max(100, max_obs_delay + RNN_SEQUENCE_LENGTH + 20)
        leader_qd_buffer_size = max(100, max_obs_delay + RNN_SEQUENCE_LENGTH + 20)

        self.leader_q_history = deque(maxlen=leader_q_buffer_size)
        self.leader_qd_history = deque(maxlen=leader_qd_buffer_size)

        # Remote state history buffers (for observation)
        self.remote_q_history = deque(maxlen=REMOTE_HISTORY_LEN)
        self.remote_qd_history = deque(maxlen=REMOTE_HISTORY_LEN)

        # Store predicted target from LSTM
        self._last_predicted_target: Optional[np.ndarray] = None
        
        # Get steps to fill the LSTM buffer
        self.buffer_fill_steps = RNN_SEQUENCE_LENGTH 
        
        # Get leader's warmup steps
        self.leader_warmup_steps = int(
            self.leader._warm_up_duration / (1.0 / self.control_freq)
        )
        
        # Total warmup is SUM of both
        self.total_warmup_phase_steps = self.leader_warmup_steps + self.buffer_fill_steps
        self.steps_remaining_in_warmup = 0

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
        
        # Reset the warmup counter
        self.steps_remaining_in_warmup = self.total_warmup_phase_steps
        
        self.episode_count += 1
        self._last_predicted_target = None  # Clear predicted target

        # Reset leader and remote robot
        leader_start_q, _ = self.leader.reset(seed=seed, options=options)
        self.remote_robot.reset(initial_qpos=self.initial_qpos)
        
        # Clear history buffers
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        
        # Clear remote state history
        self.remote_q_history.clear()
        self.remote_qd_history.clear()

        # Pre-fill leader history buffer
        max_history_needed = self.delay_simulator._obs_delay_max_steps + RNN_SEQUENCE_LENGTH + 5
        for _ in range(max_history_needed):
            self.leader_q_history.append(leader_start_q.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))
        
        return self._get_observation(), self._get_info()

    def set_predicted_target(self, predicted_target: np.ndarray) -> None:
        """predicted target: q(7 dims) + qd(7 dims)"""
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

        # Get delay input for remote robot
        delayed_q = self._get_delayed_q()

        # Warmup phase logic
        if self.steps_remaining_in_warmup > 0:
            self.steps_remaining_in_warmup -= 1
            
            target_q_for_remote = delayed_q 
            torque_compensation_for_remote = np.zeros(N_JOINTS)
            self._last_predicted_target = None
            
            step_info = self.remote_robot.step(
                target_q=target_q_for_remote,      
                torque_compensation=torque_compensation_for_remote,
            )
            
            remote_q, remote_qd = self.remote_robot.get_joint_state()
            
            reward = 0.0
            terminated = False # <<< Explicitly False during warmup
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

        # RL learning steps, or Data Collection (if _last_predicted_target is None)
        if self._last_predicted_target is not None:
            target_q_for_remote = self._last_predicted_target[:N_JOINTS]
            torque_compensation_for_remote = action
        else:
            # Data Collection Mode Fallback
            target_q_for_remote = delayed_q 
            torque_compensation_for_remote = np.zeros(N_JOINTS)
        
        step_info = self.remote_robot.step(
            target_q=target_q_for_remote,      
            torque_compensation=torque_compensation_for_remote,
        )
        
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        reward, r_tracking = self._calculate_reward(action)

        self.hist_total_reward.append(reward)
        self.hist_tracking_reward.append(r_tracking)
        true_target = self.get_true_current_target()
        true_target_q_for_plot = true_target[:N_JOINTS]
        
        # --- [MODIFICATION START] ---
        # Initialize terminated as False. Only check for it if we are in RL mode.
        terminated = False
        term_penalty = 0.0
        predicted_q = None
        joint_error = 0.0 # Initialize
        
        if self._last_predicted_target is not None:
            # --- This block now ONLY runs in RL mode ---
            predicted_q = self._last_predicted_target[:N_JOINTS]
            joint_error = np.linalg.norm(predicted_q - remote_q)
            
            terminated, term_penalty = self._check_termination(joint_error, remote_q)
            if terminated:
                reward += term_penalty
        
        true_target_q = true_target_q_for_plot # Ground truth

        # Check truncation (this is our main reset trigger)
        truncated = self.current_step >= self.max_episode_steps

        # Print out information for debugging
        if (self.current_step % 100 == 1) or (terminated): # 'terminated' will only be true in RL mode
            np.set_printoptions(precision=4, suppress=True, linewidth=120)
            
            print(f"\n[DEBUG] Step: {self.current_step}")
            print(f"  True Target q: {true_target_q}")
            if predicted_q is not None:
                print(f"  Predicted q:   {predicted_q}")
                pred_error_norm = np.linalg.norm(true_target_q - predicted_q)
                print(f"  -> Prediction Error (norm): {pred_error_norm:.4f} rad")
            else:
                print(f"  Predicted q:   None (Data Collection Mode)") # This is now expected
            
            print(f"  Remote Robot q:  {remote_q}")
            print(f"  -> Tracking Error (norm): {np.linalg.norm(true_target_q - remote_q):.4f} rad")
            if self._last_predicted_target is not None:
                print(f"  -> Joint Error (for term): {joint_error:.6f}")
        # --- [MODIFICATION END] ---
        
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
        Construct observation for RL agent with optimized structure (112D).
        
        Observation Components:
        1. remote_q (7D): Current remote robot position
        2. remote_qd (7D): Current remote robot velocity
        3. remote_q_history (35D): Position trajectory (5 timesteps × 7 joints)
        4. remote_qd_history (35D): Velocity trajectory (5 timesteps × 7 joints)
        5. predicted_q (7D): LSTM predicted position
        6. predicted_qd (7D): LSTM predicted velocity
        7. error_q (7D): Position error (predicted - remote)
        8. error_qd (7D): Velocity error (predicted - remote)
        """
        
        # Get current remote state
        remote_q, remote_qd = self.remote_robot.get_joint_state()
    
        # Get LSTM prediction (frozen, pre-trained)
        if self._last_predicted_target is not None:
            predicted_q = self._last_predicted_target[:N_JOINTS]
            predicted_qd = self._last_predicted_target[N_JOINTS:]
        else:
            predicted_q = remote_q.copy()
            predicted_qd = remote_qd.copy()

        # Add current observations to history buffers
        self.remote_q_history.append(remote_q.copy())
        self.remote_qd_history.append(remote_qd.copy())
        
        # Pad with zeros if not enough history yet
        while len(self.remote_q_history) < REMOTE_HISTORY_LEN:
            self.remote_q_history.appendleft(np.zeros(N_JOINTS))
            self.remote_qd_history.appendleft(np.zeros(N_JOINTS))
        
        # Flatten into vectors
        remote_q_history = np.concatenate(list(self.remote_q_history))  # 35D
        remote_qd_history = np.concatenate(list(self.remote_qd_history))  # 35D

        # Error signals
        error_q = predicted_q - remote_q  # 7D
        error_qd = predicted_qd - remote_qd  # 7D

        # Concatenate all components into final observation
        obs = np.concatenate([
            remote_q,           # 7D
            remote_qd,          # 7D
            remote_q_history,   # 35D
            remote_qd_history,  # 35D
            predicted_q,        # 7D
            predicted_qd,       # 7D
            error_q,            # 7D
            error_qd            # 7D
        ]).astype(np.float32)
        
        return obs
        
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
            return np.tile(initial_state, (buffer_length, 1)).flatten().astype(np.float32) # Fix: tile shape

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

    def _calculate_reward(
        self,
        action: np.ndarray,  # tau_compensation
    ) -> Tuple[float, float]: # MODIFIED: Return r_tracking for logging
        """
        Calculate dense reward combining prediction and tracking accuracy.
        """
        
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()
        true_target_q = true_target[:N_JOINTS]
        true_target_qd = true_target[N_JOINTS:]
        
        # PRIMARY OBJECTIVE: Position Tracking (Per-Joint)
        tracking_error_q_vec = true_target_q - remote_q
        r_pos_per_joint = -TRACKING_ERROR_SCALE * (tracking_error_q_vec**2)
        r_pos = np.sum(r_pos_per_joint) # Sum of individual penalties
        
        # SECONDARY OBJECTIVE: Velocity Tracking (Per-Joint)
        tracking_error_qd_vec = true_target_qd - remote_qd
        r_vel_per_joint = -VELOCITY_ERROR_SCALE * (tracking_error_qd_vec**2)
        r_vel = np.sum(r_vel_per_joint)
        
        # Combine:
        r_tracking = r_pos + r_vel
        
        # TERTIARY OBJECTIVE: Smooth Control (L2 penalty)
        action_penalty = -ACTION_PENALTY_WEIGHT * np.mean(np.square(action))
        
        # Combine all
        total_reward = r_tracking + action_penalty

        # Logging
        if self.current_step % 1000 == 0:
            # [FIX] Log the L2-norm (total error) in RADIANS for info
            tracking_error_q_norm = np.linalg.norm(tracking_error_q_vec) 
            tracking_error_qd_norm = np.linalg.norm(tracking_error_qd_vec)
            
            print(f"\n{'='*70}")
            print(f"[Reward Analysis - Step {self.current_step}]")
            print(f"{'='*70}")
            # [FIX] Change label from "mm" to "rad"
            print(f"Position Error (L2-norm): {tracking_error_q_norm:.4f} rad → r_pos = {r_pos:.4f}")
            print(f"Velocity Error (L2-norm): {tracking_error_qd_norm:.4f} rad/s → r_vel = {r_vel:.4f}")
            print(f"Combined Tracking: {r_tracking:.4f}")
            print(f"Action Magnitude (RMS): {np.sqrt(np.mean(np.square(action))):.4f} Nm → Penalty = {action_penalty:.4f}")
            print(f"TOTAL REWARD: {total_reward:.4f}")
            print(f"{'='*70}\n")
        
        # [CRITICAL FIX] Do NOT clip the reward.
        return float(total_reward), float(r_tracking)
    
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
        
        # --- [NEW] WARMUP LOGIC ---
        # Add warmup status to info
        info_dict['is_in_warmup'] = (self.steps_remaining_in_warmup > 0)
        # --- END WARMUP LOGIC ---

        return info_dict

    def render(self) -> None:
        """Render the live plot of rewards during training."""
        
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer, self.ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            self.viewer.suptitle(f'Live Teleoperation Tracking - Run {self.episode_count}', fontsize=12)
            
            # total reward
            self.line1, = self.ax[0].plot([], [], label='TOTAL Step Reward', color='green')
            self.ax[0].set_ylabel('Total Reward')
            self.ax[0].legend(loc='upper right')
            
            # tracking reward
            self.line2, = self.ax[1].plot([], [], label='Tracking Reward (Weighted)', color='blue')
            self.ax[1].set_ylabel('Tracking Reward')
            self.ax[1].set_xlabel(f'Time Steps (History Length: {self.plot_history_len})')
            self.ax[1].legend(loc='upper right')
            
            plt.ion() # Turn on interactive mode for non-blocking plot updates
            plt.show(block=False)

        # 1. Update Plot Title and X-axis Data
        x_data = np.arange(self._step_counter - len(self.hist_total_reward) + 1, 
                           self._step_counter + 1)
        
        # 2. Update Y-axis Data
        self.line1.set_data(x_data, self.hist_total_reward)
        self.line2.set_data(x_data, self.hist_tracking_reward)
        # self.line3.set_data(x_data, self.hist_prediction_reward) # Removed

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