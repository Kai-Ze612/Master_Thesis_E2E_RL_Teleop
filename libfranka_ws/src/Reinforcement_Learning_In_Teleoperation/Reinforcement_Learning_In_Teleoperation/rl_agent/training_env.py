"""
The main RL training environment with delay simulation.

Integrated with:
- DelaySimulator for realistic network delay patterns
- LocalRobotSimulator for reference joint trajectory generation
- RemoteRobotSimulator (joint space version) for follower robot dynamics
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
    DEFAULT_CONTROL_FREQ,
    JOINT_LIMIT_MARGIN,
    N_JOINTS,
    TORQUE_LIMITS,
    MAX_TORQUE_COMPENSATION,
    OBS_DIM
)

class TeleoperationEnvWithDelay(gym.Env):
    """
    Gymnasium environment for teleoperation with variable network delay.

    Architecture:
    - Leader (LocalRobotSimulator): Generates reference trajectories
    - Delay Simulator: Models network observation and action delays
    - Follower (RemoteRobotSimulator): Tracks leader with delay-adaptive control
    - RL Agent: Learns torque corrections on top of baseline controller
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        experiment_config: ExperimentConfig = ExperimentConfig.MEDIUM_DELAY,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_trajectory: bool = False,
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
        leader_q_buffer_size = max(100, max_obs_delay + self.target_history_len + 20)
        action_buffer_size = max(50, max_action_delay + self.action_history_len + 10)

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
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to the initial state for a new episode."""
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1

        leader_start_q, _ = self.leader.reset(seed=seed, options=options)
        self.remote_robot.reset(initial_qpos=self.initial_qpos)

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

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step in the environment."""
        self.current_step += 1

        new_leader_q, _, _, _, leader_info = self.leader.step()
        new_leader_qd = leader_info.get('joint_vel', np.zeros(self.n_joints))

        self.leader_q_history.append(new_leader_q.copy())
        self.leader_qd_history.append(new_leader_qd.copy())
        self.action_history.append(action.copy())

        delayed_q = self._get_delayed_q()
        delayed_action = self._get_delayed_action()

        step_info = self.remote_robot.step(
            target_q=delayed_q,
            torque_compensation=delayed_action,
        )

        remote_q, _ = self.remote_robot.get_joint_state()
        real_time_error = np.linalg.norm(new_leader_q - remote_q)

        reward = self._calculate_reward(real_time_error, action)
        terminated, term_penalty = self._check_termination(real_time_error, remote_q)
        reward += term_penalty
        truncated = self.current_step >= self.max_episode_steps

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self) -> np.ndarray:
        """Assembles the observation vector for the RL agent."""
        remote_q, remote_qd = self.remote_robot.get_joint_state()
        delayed_target_q = self._get_delayed_q()

        target_q_history = np.array(list(self.leader_q_history)[-self.target_history_len:]).flatten()
        target_qd_history = np.array(list(self.leader_qd_history)[-self.target_history_len:]).flatten()

        obs_delay_steps = self.delay_simulator.get_observation_delay_steps(len(self.leader_q_history))
        delay_magnitude = np.array([obs_delay_steps / 100.0])

        action_history = np.array(list(self.action_history)[-self.action_history_len:]).flatten()

        return np.concatenate([
            remote_q,
            remote_qd,
            delayed_target_q,
            target_q_history,
            target_qd_history,
            delay_magnitude,
            action_history
        ]).astype(np.float32)

    def _calculate_reward(self, joint_error: float, action: np.ndarray) -> float:
        """Calculates the reward for the current step."""
        tracking_reward = np.exp(-10.0 * joint_error**2)
        action_penalty = -0.01 * np.sum(np.square(action))
        return tracking_reward + action_penalty
    
    def _check_termination(self, joint_error: float, remote_q: np.ndarray) -> Tuple[bool, float]:
        """Checks if the episode should terminate."""
        at_limits = np.any(remote_q <= self.joint_limits_lower + self.joint_limit_margin) or \
                    np.any(remote_q >= self.joint_limits_upper - self.joint_limit_margin)
        high_error = joint_error > self.max_joint_error
        terminated = at_limits or high_error
        return terminated, -10.0 if terminated else 0.0

    def _get_info(self) -> Dict[str, Any]:
        """Returns diagnostic information for logging."""
        remote_q, _ = self.remote_robot.get_joint_state()
        real_time_error = np.linalg.norm(self.leader_q_history[-1] - remote_q)
        return {'real_time_joint_error': real_time_error}

    def render(self) -> None:
        """Rendering is not implemented for this environment."""
        pass

    def close(self) -> None:
        """Performs any necessary cleanup."""
        pass
