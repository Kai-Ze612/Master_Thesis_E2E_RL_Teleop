"""
The main RL environment (Gymnasium) with delay simulation.

Architecture:
- Agent lives on the LOCAL side.
- Leader (LocalRobotSimulator): Generates reference trajectories (real-time to agent).
- Delay Simulator:
    - Models Action Delay (alpha) for commands (Agent -> Remote).
    - Models Observation Delay (omega) for state/reward (Remote -> Agent).
- Follower (RemoteRobotSimulator): Executes delayed commands.
- RL Agent: Learns torque corrections.

"""

# RL library imports
import gymnasium as gym
from gymnasium import spaces

# Python standard libraries
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional

# Custom imports
from .local_robot_simulator import LocalRobotSimulator, TrajectoryType
from .remote_robot_simulator import RemoteRobotSimulator
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import (
    DelaySimulator,
    ExperimentConfig
)

from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    INITIAL_JOINT_CONFIG,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    MAX_EPISODE_STEPS,
    MAX_JOINT_ERROR_TERMINATION,
    ACTION_HISTORY_LEN,
    TARGET_HISTORY_LEN,
    OBS_HISTORY_LEN,
    DEFAULT_CONTROL_FREQ,
    JOINT_LIMIT_MARGIN,
    N_JOINTS,
    TORQUE_LIMITS,
    MAX_TORQUE_COMPENSATION,
    OBS_PACKET_FEATURE_SIZE,
    OBS_DIM,  # Optional: for reference in "full" mode
)


class TeleoperationEnvWithDelay(gym.Env):

    metadata = {'render_modes': []}

    def __init__(
        self,
        experiment_config: ExperimentConfig = ExperimentConfig.MEDIUM_DELAY,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_trajectory: bool = False,
        obs_mode: str = "minimal",
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
        self.obs_history_len = OBS_HISTORY_LEN

        max_obs_delay = self.delay_simulator._obs_delay_max_steps
        max_action_delay = self.delay_simulator._action_delay_max_steps
        
        # Buffer for real-time leader (Local to Agent)
        leader_buffer_size = max(100, max_action_delay + self.target_history_len + 20)
        self.leader_q_history = deque(maxlen=leader_buffer_size)
        self.leader_qd_history = deque(maxlen=leader_buffer_size)
        
        # Buffer for agent's action (Agent to Remote) 
        action_buffer_size = max(100, max_action_delay + self.action_history_len + 10)
        self.action_history = deque(maxlen=action_buffer_size)

        # Buffer for ground truth remote states(Remote to Agent)
        remote_buffer_size = max(100, max_obs_delay + 20)
        self.remote_q_history = deque(maxlen=remote_buffer_size)
        self.remote_qd_history = deque(maxlen=remote_buffer_size)
        self.reward_history = deque(maxlen=remote_buffer_size)
        self.terminated_history = deque(maxlen=remote_buffer_size)

        # Buffer for delay remote states (Remote to Agent)
        self.obs_packet_history = deque(maxlen=self.obs_history_len)
        self.obs_packet_feature_size = self.n_joints + self.n_joints + 1
        
        # State variables for current step delays
        self.current_obs_delay_steps = 0
        self.current_action_delay_steps = 0
        
        # Store observation mode
        self.obs_mode = obs_mode  # ← ADD THIS LINE
        
        # Action and observation spaces setup
        self.action_space = spaces.Box(
            low=-MAX_TORQUE_COMPENSATION.copy(),
            high=MAX_TORQUE_COMPENSATION.copy(),
            shape=(self.n_joints,),
            dtype=np.float32
        )
        
        # Calculate observation dimension dynamically based on mode
        obs_dim = self._get_observation_dim()  # ← CHANGE THIS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
    
    def _get_observation_dim(self) -> int:
        """Calculate observation dimension based on mode."""
        if self.obs_mode == "minimal":
            return (
                self.n_joints +     # delayed_remote_q
                self.n_joints +     # delayed_remote_qd
                self.n_joints +     # realtime_target_q
                self.n_joints +     # realtime_target_qd
                self.n_joints +     # tracking_error
                self.n_joints +     # velocity_error
                self.n_joints +     # recent_action
                1 +                 # obs_delay_magnitude
                1                   # act_delay_magnitude
            )  # Total: 51
            
        elif self.obs_mode == "context":
            return (
                self.n_joints +     # delayed_remote_q
                self.n_joints +     # delayed_remote_qd
                self.n_joints +     # realtime_target_q
                self.n_joints +     # realtime_target_qd
                self.n_joints +     # tracking_error
                self.n_joints +     # velocity_error
                self.n_joints +     # recent_action
                1 +                 # obs_delay_magnitude
                1 +                 # act_delay_magnitude
                self.n_joints +     # prev_remote_q
                self.n_joints +     # prev_remote_qd
                self.n_joints * 2 + # target_q_short (last 2)
                self.n_joints * 2   # action_short (last 2)
            )  # Total: 93
            
        else:  # "full"
            return (
                self.n_joints +
                self.n_joints +
                1 +
                self.n_joints +
                (self.n_joints * self.target_history_len) +
                (self.n_joints * self.target_history_len) +
                1 +
                (self.n_joints * self.action_history_len) +
                (OBS_PACKET_FEATURE_SIZE * self.obs_history_len)
            )  # Total: 243

    def _get_observation_minimal(self) -> np.ndarray:
        """Minimal observation space (51 dims) - RECOMMENDED."""
        # Current delayed state
        most_recent_packet = self.obs_packet_history[-1]
        delayed_remote_q = most_recent_packet[:self.n_joints]
        delayed_remote_qd = most_recent_packet[self.n_joints:2*self.n_joints]
        obs_delay_magnitude = most_recent_packet[-1:]
        
        # Current real-time target
        realtime_target_q = self.leader_q_history[-1].copy()
        realtime_target_qd = self.leader_qd_history[-1].copy()
        
        # Explicit error signals (KEY: helps learning)
        tracking_error = realtime_target_q - delayed_remote_q
        velocity_error = realtime_target_qd - delayed_remote_qd
        
        # Recent action and delay
        recent_action = self.action_history[-1].copy()
        act_delay_magnitude = np.array([self.current_action_delay_steps / 100.0])
        
        return np.concatenate([
            delayed_remote_q,        # 7
            delayed_remote_qd,       # 7
            realtime_target_q,       # 7
            realtime_target_qd,      # 7
            tracking_error,          # 7 ← Explicit error signal
            velocity_error,          # 7 ← Velocity mismatch
            recent_action,           # 7
            obs_delay_magnitude,     # 1
            act_delay_magnitude,     # 1
        ]).astype(np.float32)

    def _get_observation_with_context(self) -> np.ndarray:
        """Observation with short-term context (93 dims)."""
        # Get minimal observation components
        most_recent_packet = self.obs_packet_history[-1]
        delayed_remote_q = most_recent_packet[:self.n_joints]
        delayed_remote_qd = most_recent_packet[self.n_joints:2*self.n_joints]
        obs_delay_magnitude = most_recent_packet[-1:]
        
        realtime_target_q = self.leader_q_history[-1].copy()
        realtime_target_qd = self.leader_qd_history[-1].copy()
        
        tracking_error = realtime_target_q - delayed_remote_q
        velocity_error = realtime_target_qd - delayed_remote_qd
        
        recent_action = self.action_history[-1].copy()
        act_delay_magnitude = np.array([self.current_action_delay_steps / 100.0])
        
        # Add short-term context
        if len(self.obs_packet_history) >= 2:
            prev_packet = self.obs_packet_history[-2]
            prev_remote_q = prev_packet[:self.n_joints]
            prev_remote_qd = prev_packet[self.n_joints:2*self.n_joints]
        else:
            prev_remote_q = delayed_remote_q.copy()
            prev_remote_qd = delayed_remote_qd.copy()
        
        # Last 2 target positions
        target_q_short = np.array(list(self.leader_q_history)[-2:]).flatten()
        
        # Last 2 actions
        action_short = np.array(list(self.action_history)[-2:]).flatten()
        
        return np.concatenate([
            delayed_remote_q,        # 7
            delayed_remote_qd,       # 7
            realtime_target_q,       # 7
            realtime_target_qd,      # 7
            tracking_error,          # 7
            velocity_error,          # 7
            recent_action,           # 7
            obs_delay_magnitude,     # 1
            act_delay_magnitude,     # 1
            prev_remote_q,           # 7
            prev_remote_qd,          # 7
            target_q_short,          # 14
            action_short,            # 14
        ]).astype(np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to the initial state for a new episode."""
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1

        self.current_obs_delay_steps = 0
        self.current_action_delay_steps = 0
        
        # Reset leader (real-time)
        leader_start_q, _ = self.leader.reset(seed=seed, options=options)
        
        # Reset remote robot
        remote_start_q = self.initial_qpos.copy()
        self.remote_robot.reset(initial_qpos=remote_start_q)

        # Clear all history buffers
        self.leader_q_history.clear()
        self.leader_qd_history.clear()
        self.action_history.clear()
        self.remote_q_history.clear()
        self.remote_qd_history.clear()
        self.reward_history.clear()
        self.terminated_history.clear()
        self.obs_packet_history.clear()

        # Pre-fill real-time leader and action buffers
        leader_buffer_len = self.delay_simulator._action_delay_max_steps + self.target_history_len + 5
        for _ in range(leader_buffer_len):
            self.leader_q_history.append(leader_start_q.copy())
            self.leader_qd_history.append(np.zeros(self.n_joints))
            
        action_buffer_len = self.delay_simulator._action_delay_max_steps + self.action_history_len + 5
        for _ in range(action_buffer_len):
            self.action_history.append(np.zeros(self.n_joints))

        # Pre-fill ground truth remote state and reward buffers
        remote_buffer_len = self.delay_simulator._obs_delay_max_steps + 5
        for _ in range(remote_buffer_len):
            self.remote_q_history.append(remote_start_q.copy())
            self.remote_qd_history.append(np.zeros(self.n_joints))
            self.reward_history.append(0.0)
            self.terminated_history.append(False)

        # Pre-fill the agent's observation history
        initial_obs_packet = np.concatenate([
            remote_start_q.copy(),
            np.zeros(self.n_joints),
            np.array([0.0]) # Initial delay magnitude
        ])
        for _ in range(self.obs_history_len):
            self.obs_packet_history.append(initial_obs_packet)
        
        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray # This is action a(t)
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step in the environment."""
        self.current_step += 1

        ## Local Robot to Agent 
        # Agent sends action a(t)
        self.action_history.append(action.copy())
        new_leader_q, _, _, _, leader_info = self.leader.step()
        new_leader_qd = leader_info.get('joint_vel', np.zeros(self.n_joints))
        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())

        ## Simulation Action Delay (Agent -> Remote)
        self.current_action_delay_steps = self.delay_simulator.get_action_delay_steps()
        act_buffer_len = len(self.action_history)
        act_delay_index = min(self.current_action_delay_steps, act_buffer_len - 1)
        delayed_action = self.action_history[-act_delay_index - 1].copy()
        
        leader_buffer_len = len(self.leader_q_history)
        goal_delay_index = min(self.current_action_delay_steps, leader_buffer_len - 1)
        delayed_target_q = self.leader_q_history[-goal_delay_index - 1].copy()

        ## Remote Robot
        step_info = self.remote_robot.step(
            target_q=delayed_target_q,
            torque_compensation=delayed_action,
        )
        
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        self.remote_q_history.append(remote_q.copy())
        self.remote_qd_history.append(remote_qd.copy())
        
        # CALCULATE REAL-TIME REWARD/TERMINATION (at Remote)
        real_time_error = np.linalg.norm(new_leader_q - remote_q)
        real_time_reward = self._calculate_reward(real_time_error, delayed_action) 
        real_time_terminated, term_penalty = self._check_termination(real_time_error, remote_q)
        
        self.reward_history.append(real_time_reward + term_penalty)
        self.terminated_history.append(real_time_terminated)
        
        # SIMULATE OBSERVATION DELAY (Remote -> Agent)
        self.current_obs_delay_steps = self.delay_simulator.get_observation_delay_steps(len(self.reward_history))
        obs_buffer_len = len(self.reward_history)
        obs_delay_index = min(self.current_obs_delay_steps, obs_buffer_len - 1)
        
        delayed_reward = self.reward_history[-obs_delay_index - 1]
        delayed_terminated = self.terminated_history[-obs_delay_index - 1]

        # Get the observation packet that just arrived
        delayed_remote_q = self.remote_q_history[-obs_delay_index - 1].copy()
        delayed_remote_qd = self.remote_qd_history[-obs_delay_index - 1].copy()
        obs_delay_magnitude = np.array([self.current_obs_delay_steps / 100.0])
        
        # Store this packet in the agent's observable history
        new_obs_packet = np.concatenate([
            delayed_remote_q,
            delayed_remote_qd,
            obs_delay_magnitude
        ])
        self.obs_packet_history.append(new_obs_packet)
        
        # Return to Agent
        truncated = self.current_step >= self.max_episode_steps
        
        return (
            self._get_observation(),
            delayed_reward,
            delayed_terminated,
            truncated,
            self._get_info(real_time_error) # Info packet contains ground truth
        )

    def _get_observation_full(self) -> np.ndarray:  # ← RENAME from _get_observation
        """Full observation with all history (243 dims) - ORIGINAL IMPLEMENTATION."""
        
        # Get MOST RECENT Delayed Remote State
        most_recent_packet = self.obs_packet_history[-1]
        delayed_remote_q = most_recent_packet[:self.n_joints]
        delayed_remote_qd = most_recent_packet[self.n_joints : 2*self.n_joints]
        obs_delay_magnitude = most_recent_packet[-1:]

        # Get Real-Time Leader State
        realtime_target_q = self.leader_q_history[-1].copy()

        # Get Real-Time Histories
        target_q_history = np.array(list(self.leader_q_history)[-self.target_history_len:]).flatten()
        target_qd_history = np.array(list(self.leader_qd_history)[-self.target_history_len:]).flatten()
        action_history = np.array(list(self.action_history)[-self.action_history_len:]).flatten()
        
        # Get Observation History
        obs_history = np.array(list(self.obs_packet_history)).flatten()

        # Get CURRENT Action Delay Magnitude
        act_delay_magnitude = np.array([self.current_action_delay_steps / 100.0])

        return np.concatenate([
            delayed_remote_q,
            delayed_remote_qd,
            obs_delay_magnitude,
            realtime_target_q,
            target_q_history,
            target_qd_history,
            act_delay_magnitude,
            action_history,
            obs_history
        ]).astype(np.float32)

    def _calculate_reward(self, joint_error: float, executed_action: np.ndarray) -> float:
        """
        Calculates the (real-time) reward for the current step.
        UPDATED: Quadratic reward for better gradient signal.
        """
        # Quadratic tracking reward (provides gradient across all error ranges)
        tracking_reward = -10.0 * joint_error**2
        
        # Action penalty (prevent excessive torques)
        action_penalty = -0.001 * np.sum(np.square(executed_action))
        
        # Action smoothness penalty (reduce vibration)
        if len(self.action_history) >= 2:
            action_diff = executed_action - self.action_history[-2]
            smoothness_penalty = -0.001 * np.sum(np.square(action_diff))
        else:
            smoothness_penalty = 0.0
        
        return tracking_reward + action_penalty + smoothness_penalty
    
    def _check_termination(self, joint_error: float, remote_q: np.ndarray) -> Tuple[bool, float]:
        """Checks if the (real-time) episode should terminate."""
        at_limits = np.any(remote_q <= self.joint_limits_lower + self.joint_limit_margin) or \
                    np.any(remote_q >= self.joint_limits_upper - self.joint_limit_margin)
        high_error = joint_error > self.max_joint_error
        terminated = at_limits or high_error
        return terminated, -10.0 if terminated else 0.0

    # AFTER
    def _get_info(self, real_time_error: float = 0.0) -> Dict[str, Any]:
        """Returns diagnostic information for logging (ground truth)."""
        return {
            'real_time_joint_error': real_time_error,
            'obs_delay_steps': self.current_obs_delay_steps,
            'action_delay_steps': self.current_action_delay_steps,
        }

    def render(self) -> None:
        """Rendering is not implemented for this environment."""
        pass

    def close(self) -> None:
        """Performs any necessary cleanup."""
        pass
