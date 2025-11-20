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
    TRAJECTORY_FREQUENCY,
    WARM_UP_DURATION,
    DELAY_INPUT_NORM_FACTOR,
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
    ):
        super().__init__()
        
        # Rendering setup
        self.render_mode = render_mode
        self.viewer = None
        self.ax = None
        self.plot_history_len = 1000
        
        # Import configurations
        self.hist_tracking_reward = deque(maxlen=self.plot_history_len)
        self.hist_total_reward = deque(maxlen=self.plot_history_len)
        self._step_counter = 0
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.control_freq = DEFAULT_CONTROL_FREQ
        self.current_step = 0
        self.episode_count = 0
        self.max_joint_error = MAX_JOINT_ERROR_TERMINATION
        self.joint_limit_margin = JOINT_LIMIT_MARGIN
        self.n_joints = N_JOINTS
        self.initial_qpos = INITIAL_JOINT_CONFIG.copy()
        self.joint_limits_lower = JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = JOINT_LIMITS_UPPER.copy()
        self.torque_limits = TORQUE_LIMITS.copy()
        self.delay_config = delay_config
        
        # Initialize delay simulator
        self.delay_simulator = DelaySimulator(control_freq=self.control_freq, config=delay_config, seed=seed)
        
        # Initialize local robot
        self.trajectory_type = trajectory_type
        self.randomize_trajectory = randomize_trajectory
        self.leader = LocalRobotSimulator(trajectory_type=self.trajectory_type, randomize_params=self.randomize_trajectory)
        
        # Initialize remote robot
        self.remote_robot = RemoteRobotSimulator(delay_config=delay_config, seed=seed)
        
        # Initialize observation delay
        max_obs_delay = self.delay_simulator._obs_delay_max_steps
        
        leader_q_buffer_size = max(100, max_obs_delay + RNN_SEQUENCE_LENGTH + 20)
        leader_qd_buffer_size = max(100, max_obs_delay + RNN_SEQUENCE_LENGTH + 20)
        self.leader_q_history = deque(maxlen=leader_q_buffer_size)
        self.leader_qd_history = deque(maxlen=leader_qd_buffer_size)
        self.remote_q_history = deque(maxlen=REMOTE_HISTORY_LEN)
        self.remote_qd_history = deque(maxlen=REMOTE_HISTORY_LEN)
        self._last_predicted_target: Optional[np.ndarray] = None
        self.buffer_fill_steps = RNN_SEQUENCE_LENGTH 
        
        # Warm-up phase parameters
        self.warmup_time = WARM_UP_DURATION 
        self.leader_warmup_steps = int(self.warmup_time * self.control_freq)
        self.total_warmup_phase_steps = self.leader_warmup_steps + self.buffer_fill_steps
        self.steps_remaining_in_warmup = 0

        
        # 1. Torque Bounds (from config)
        torque_low = -MAX_TORQUE_COMPENSATION.copy()
        torque_high = MAX_TORQUE_COMPENSATION.copy()
        
        # 2. Prediction Bounds (Unbounded / Infinity)
        # Predictions can be anywhere in state space, so we use +/- infinity
        pred_low = np.full(self.n_joints * 2, -np.inf, dtype=np.float32)
        pred_high = np.full(self.n_joints * 2, np.inf, dtype=np.float32)
        
        # 3. Concatenate to form 21-dim bounds
        action_low = np.concatenate([torque_low, pred_low])
        action_high = np.concatenate([torque_high, pred_high])

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high, 
            shape=(self.n_joints * 3,), # 7 (Torque) + 14 (State) = 21
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        # Internal tick counter
        self._current_tick = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1
        self._last_predicted_target = None
        
        self._current_tick = 0
        
        leader_start_q, _ = self.leader.reset(seed=seed, options=options)
        self.remote_robot.reset(initial_qpos=self.initial_qpos)
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        self.remote_q_history.clear()
        self.remote_qd_history.clear()
        
        max_history_needed = self.delay_simulator._obs_delay_max_steps + RNN_SEQUENCE_LENGTH + 5
        for _ in range(max_history_needed):
            self.leader_q_history.append(leader_start_q.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))
        
        start_target_q = self._get_delayed_q()
        self.remote_robot.reset(initial_qpos=start_target_q)
        self.remote_q_history.clear()
        self.remote_qd_history.clear()
        for _ in range(REMOTE_HISTORY_LEN):
            self.remote_q_history.append(start_target_q.copy())
            self.remote_qd_history.append(np.zeros(self.n_joints))
            
        self.steps_remaining_in_warmup = 0
        return self._get_observation(), self._get_info()

    def set_predicted_target(self, predicted_target: np.ndarray) -> None:
        """predicted target: q(7 dims) + qd(7 dims)"""
        if predicted_target.shape[0] != N_JOINTS * 2:
            raise ValueError(f"predicted_target must have shape ({N_JOINTS * 2},), got {predicted_target.shape}")
        
        self._last_predicted_target = predicted_target.copy()

    def get_current_observation_delay(self) -> int:
        history_len = len(self.leader_q_history)
        sim_delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
        return int(sim_delay_steps)

    def _get_delayed_q(self) -> np.ndarray:
        buffer_len = len(self.leader_q_history)
        delay_steps = self.get_current_observation_delay() # Uses integer ticks
        delay_index = min(delay_steps, buffer_len - 1)
        return self.leader_q_history[-delay_index - 1].copy()
    
    def _get_delayed_qd(self) -> np.ndarray:
        buffer_len = len(self.leader_qd_history)
        delay_steps = self.get_current_observation_delay()
        delay_index = min(delay_steps, buffer_len - 1)
        return self.leader_qd_history[-delay_index - 1].copy()

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep in the environment.
        Action can be 7D (Torque only) or 21D (Torque + Predicted State).
        """
        self._current_tick += 1
        
        # Action parsing
        if action.shape[0] == self.n_joints * 3: # 21 Dimensions
            # 1. Extract Torque (First 7)
            actual_action = action[:self.n_joints]
            
            # 2. Extract Prediction (Last 14)
            predicted_state = action[self.n_joints:]
            self.set_predicted_target(predicted_state)
        else:
            # Fallback for 7-dim action (RL only, no prediction update)
            actual_action = action
            # In this case, self._last_predicted_target remains whatever it was previously
        
        self.current_step += 1
        self._step_counter += 1
        
        # Update Leader (Ground Truth Trajectory)
        new_leader_q, new_leader_qd, _, _, _, _ = self.leader.step()
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())

        delayed_q = self._get_delayed_q()
        delayed_qd = self._get_delayed_qd()

        # Warmup Logic
        if self.steps_remaining_in_warmup > 0:
            self.steps_remaining_in_warmup -= 1
            target_q_for_remote = delayed_q 
            target_qd_for_remote = delayed_qd
            torque_compensation_for_remote = np.zeros(N_JOINTS)
            self._last_predicted_target = None
            step_info = self.remote_robot.step(target_q_for_remote, target_qd_for_remote, torque_compensation_for_remote)
            remote_q, remote_qd = self.remote_robot.get_joint_state()
            if self.render_mode == "human": self.render()
            return self._get_observation(), 0.0, False, self.current_step >= self.max_episode_steps, self._get_info()

        # Determine target for remote robot
        if self._last_predicted_target is not None:
            target_q_for_remote = self._last_predicted_target[:N_JOINTS]
            target_qd_for_remote = self._last_predicted_target[N_JOINTS:] 
            torque_compensation_for_remote = actual_action
        else:
            # Fallback if prediction missing (e.g. Data Collection Mode)
            target_q_for_remote = delayed_q
            target_qd_for_remote = delayed_qd
            torque_compensation_for_remote = np.zeros(N_JOINTS)
        
        step_info = self.remote_robot.step(target_q_for_remote, target_qd_for_remote, torque_compensation_for_remote)
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        
        reward, r_tracking = self._calculate_reward(actual_action)
        self.hist_total_reward.append(reward)
        self.hist_tracking_reward.append(r_tracking)
        
        true_target = self.get_true_current_target()
        terminated = False
        term_penalty = 0.0
        predicted_q = None
        joint_error = 0.0
        
        if self._last_predicted_target is not None:
            predicted_q = self._last_predicted_target[:N_JOINTS]
            joint_error = np.linalg.norm(predicted_q - remote_q)
            terminated, term_penalty = self._check_termination(joint_error, remote_q)
            if terminated: reward += term_penalty
        
        truncated = self.current_step >= self.max_episode_steps # Episode length truncation
        
        #############################################################################################
        # Debug logging 
        if (self.current_step % 1000 == 1) or (terminated):
            
            np.set_printoptions(precision=4, suppress=True, linewidth=120)
            true_target_q = true_target[:self.n_joints]
              
            print(f"\n[DEBUG] Step: {self.current_step}")
            print(f"  True Target q: {true_target_q}")
            if predicted_q is not None:
                print(f"  Predicted q:   {predicted_q}")
                pred_error_norm = np.linalg.norm(true_target_q - predicted_q)
                print(f"  -> Prediction Error (norm): {pred_error_norm:.4f} rad")
            else:
                print(f"  Predicted q:   None (Data Collection Mode)")
            
            print(f"  Remote Robot q:  {remote_q}")
            print(f"  -> Tracking Error (norm): {np.linalg.norm(true_target_q - remote_q):.4f} rad")
            if self._last_predicted_target is not None:
                print(f"  -> Joint Error (for term): {joint_error:.6f}")

            tau_pd = step_info.get('tau_pd', np.zeros(7))
            tau_rl = step_info.get('tau_rl', np.zeros(7))
            tau_total = step_info.get('tau_total', np.zeros(7))

            print(f"  Torque Breakdown (Nm):")
            print(f"   > PD Baseline:   {tau_pd}")
            print(f"   > RL Compensat:  {tau_rl}")
            print(f"   > Total Desired: {tau_total}")
        #############################################################################################
        
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
        Construct observation for RL agent (112D).
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
        
        # Add Normalized delay info
        current_delay_steps = float(self.get_current_observation_delay())
        current_delay = current_delay_steps / DELAY_INPUT_NORM_FACTOR
        
        # Concatenate all components into final observation
        obs = np.concatenate([
            remote_q,           # 7D
            remote_qd,          # 7D
            remote_q_history,   # 35D
            remote_qd_history,  # 35D
            predicted_q,        # 7D
            predicted_qd,       # 7D
            error_q,            # 7D
            error_qd,           # 7D
            [current_delay]     # 1D
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
        history_len = len(self.leader_q_history)
        
        if history_len == 0:
            initial_state = np.concatenate([
                self.initial_qpos, 
                np.zeros(self.n_joints), 
                [0.0]
            ])
            return np.tile(initial_state, (buffer_length, 1)).flatten().astype(np.float32)

        # 1. Get Raw Integer Steps for INDEXING
        raw_delay_steps = int(self.get_current_observation_delay())
                
        # 2. Get Normalized Float for LSTM Input
        normalized_delay = float(raw_delay_steps) / DELAY_INPUT_NORM_FACTOR

        # 3. Calculate Integer Indices
        # We look back 'raw_delay_steps' into the past
        most_recent_delayed_idx = -(raw_delay_steps + 1)
        oldest_idx = most_recent_delayed_idx - buffer_length + 1
        
        buffer_seq = []
        
        # 4. Loop using INTEGER indices
        for i in range(oldest_idx, most_recent_delayed_idx + 1):
            safe_idx = np.clip(i, -history_len, -1)
            
            step_vector = np.concatenate([
                self.leader_q_history[safe_idx],
                self.leader_qd_history[safe_idx],
                [normalized_delay]  # <--- USE NORMALIZED VALUE HERE
            ])
            buffer_seq.append(step_vector)
        
        buffer = np.array(buffer_seq).flatten()
        
        if buffer.shape[0] != buffer_length * (self.n_joints * 2 + 1):
            warnings.warn(f"Delayed buffer shape mismatch: got {buffer.shape[0]}")
            
        return buffer.astype(np.float32)
        
    def get_remote_state(self) -> np.ndarray:
        """Get current remote robot state (real-time, no delay)."""
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        return np.concatenate([remote_q, remote_qd]).astype(np.float32)
    
    def get_current_observation_delay(self) -> int:
        """Get the current observation delay in timesteps."""
        history_len = len(self.leader_q_history)
        return self.delay_simulator.get_observation_delay_steps(history_len)

    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, float]:
        """
        Calculate reward based on tracking error and action penalty.
        """
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        true_target = self.get_true_current_target()
        true_target_q = true_target[:N_JOINTS]
        true_target_qd = true_target[N_JOINTS:]
        tracking_error_q_vec = true_target_q - remote_q
        r_pos_per_joint = -TRACKING_ERROR_SCALE * (tracking_error_q_vec**2)
        r_pos = np.sum(r_pos_per_joint)
        tracking_error_qd_vec = true_target_qd - remote_qd
        r_vel_per_joint = -VELOCITY_ERROR_SCALE * (tracking_error_qd_vec**2)
        r_vel = np.sum(r_vel_per_joint)
        r_tracking = r_pos + r_vel
        action_penalty = -ACTION_PENALTY_WEIGHT * np.mean(np.square(action))
        total_reward = r_tracking + action_penalty
        return float(total_reward), float(r_tracking)
    
    def _check_termination(self, joint_error: float, remote_q: np.ndarray) -> Tuple[bool, float]:
        """Terminate if joint limits exceeded or joint error too high."""
        if not np.all(np.isfinite(remote_q)): return True, -100.0
        at_limits = (np.any(remote_q <= self.joint_limits_lower + self.joint_limit_margin) or np.any(remote_q >= self.joint_limits_upper - self.joint_limit_margin))
        high_error = np.isnan(joint_error) or joint_error > self.max_joint_error
        terminated = at_limits or high_error
        penalty = -10.0 if terminated else 0.0
        return terminated, penalty

    def _get_info(self) -> Dict[str, Any]:
        """
        info for debugging and analysis.
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

        info_dict['is_in_warmup'] = (self.steps_remaining_in_warmup > 0)

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