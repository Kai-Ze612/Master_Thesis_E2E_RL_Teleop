"""
The main RL training environment with delay simulation.

Integrated with:
- AdaptivePDController for delay-aware gain scheduling
- IKSolver for trajectory-continuous inverse kinematics
- DelaySimulator for realistic network delay patterns
- LocalRobotSimulator for reference trajectory generation
"""

# RL library imports
import gymnasium as gym
from gymnasium import spaces

# Python standard libraries
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional

# Custom imports
from local_robot_simulator import LocalRobotSimulator, TrajectoryType
from remote_robot_simulator import RemoteRobotSimulator
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import (
    DelaySimulator,
    ExperimentConfig
)


class TeleoperationEnvWithDelay(gym.Env):
    """
    Gymnasium environment for teleoperation with variable network delay.

    Architecture:
    - Leader (LocalRobotSimulator): Generates reference trajectories
    - Delay Simulator: Models network observation and action delays
    - Follower (RemoteRobotSimulator): Tracks leader with delay-adaptive control
    - RL Agent: Learns torque corrections on top of baseline controller

    Control Pipeline:
    1. Leader generates target position + velocity
    2. Observation delay applied to target
    3. RL agent observes delayed state + target history
    4. Action delay applied to RL corrections
    5. RemoteRobotSimulator executes:
        - IK: Cartesian target → Joint space
        - Adaptive PD: Compute desired acceleration (gains adapt to delay)
        - Inverse Dynamics: Compute baseline torque
        - RL Correction: Multiplicative torque adjustment
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        model_path: str,
        experiment_config: ExperimentConfig = ExperimentConfig.MEDIUM_DELAY,
        max_episode_steps: int = 1000,
        control_freq: int = 500,
        max_cartesian_error: float = 1.0,
        # Trajectory parameters
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_trajectory: bool = False,
        # Adaptive PD parameters
        min_gain_ratio: float = 0.3,
        delay_threshold: float = 0.2,
        # IK parameters
        max_joint_change: float = 0.1,
        continuity_gain: float = 0.5,
        # Observation history parameters
        joint_history_len: int = 1,
        action_history_len: int = 5,
        target_history_len: int = 10,
    ):
        """
        Initialize teleoperation environment.

        Args:
            model_path: Path to MuJoCo XML model file
            experiment_config: Delay configuration (LOW/MEDIUM/HIGH/NO_DELAY)
            max_episode_steps: Maximum steps per episode
            control_freq: Control frequency in Hz
            max_cartesian_error: Maximum allowed Cartesian tracking error (m)

            # Trajectory configuration
            trajectory_type: Type of reference trajectory
            randomize_trajectory: Whether to randomize trajectory parameters

            # Controller parameters
            min_gain_ratio: Minimum gain ratio for adaptive PD (at high delay)
            delay_threshold: Delay threshold for gain adaptation (seconds)

            # IK parameters
            max_joint_change: Maximum joint change per step (rad)
            continuity_gain: Null-space projection gain for trajectory continuity

            # History parameters
            joint_history_len: Length of joint state history
            action_history_len: Length of action history
            target_history_len: Length of target position history (for NN predictor)
        """

        # ============================================================
        # Basic Configuration
        # ============================================================
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        self.current_step = 0
        self.episode_count = 0

        # Safety limits
        self.max_cartesian_error = max_cartesian_error
        self.joint_limit_margin = 0.05  # Safety margin (rad)

        # ============================================================
        # Robot Configuration (Franka Panda)
        # ============================================================
        self.n_joints = 7
        self.initial_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.joint_limits_lower = np.array([
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
        ])
        self.joint_limits_upper = np.array([
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
        ])
        self.torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

        # ============================================================
        # Initialize Delay Simulator
        # ============================================================
        self.experiment_config = experiment_config
        self.delay_simulator = DelaySimulator(
            control_freq=control_freq,
            config=experiment_config,
            seed=None  # Will be set via env.reset(seed=...)
        )

        # ============================================================
        # Initialize Leader Robot (Reference Trajectory Generator)
        # ============================================================
        self.trajectory_type = trajectory_type
        self.randomize_trajectory = randomize_trajectory

        self.leader = LocalRobotSimulator(
            control_freq=control_freq,
            trajectory_type=trajectory_type,
            randomize_params=randomize_trajectory
        )

        # ============================================================
        # Initialize Remote Robot (Follower with Integrated Controllers)
        # ============================================================
        self.remote_robot = RemoteRobotSimulator(
            model_path=model_path,
            control_freq=control_freq,
            torque_limits=self.torque_limits,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper,
            # Adaptive PD parameters
            kp_nominal=None,  # Use Franka defaults
            kd_nominal=None,
            min_gain_ratio=min_gain_ratio,
            delay_threshold=delay_threshold,
            # IK parameters
            jacobian_damping=1e-4,
            max_joint_change=max_joint_change,
            continuity_gain=continuity_gain,
        )

        # ============================================================
        # [FIXED] Print Configuration AFTER Robots are Initialized
        # ============================================================
        self._print_configuration()


        # ============================================================
        # History Buffer Configuration
        # ============================================================
        self.joint_history_len = joint_history_len
        self.action_history_len = action_history_len
        self.target_history_len = target_history_len

        # Calculate required buffer sizes based on maximum possible delay
        max_obs_delay = self.delay_simulator._obs_delay_max_steps
        max_action_delay = self.delay_simulator._action_delay_steps

        # Buffers need to store: max_delay + history_length + safety_margin
        position_buffer_size = max(100, max_obs_delay + target_history_len + 20)
        action_buffer_size = max(50, max_action_delay + action_history_len + 10)

        # Initialize buffers
        self.position_history = deque(maxlen=position_buffer_size)
        self.velocity_history = deque(maxlen=position_buffer_size)
        self.action_history = deque(maxlen=action_buffer_size)
        self.joint_pos_history = deque(maxlen=joint_history_len)
        self.joint_vel_history = deque(maxlen=joint_history_len)
        self.target_position_history = deque(maxlen=target_history_len)

        # ============================================================
        # Define Action and Observation Spaces
        # ============================================================

        # Action space: normalized torque corrections in [-1, 1] → [-50%, +50%]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32
        )

        # Observation space components:
        obs_dim = (
            7 +                     # joint_pos
            7 +                     # joint_vel
            7 +                     # gravity_torque
            3 +                     # delayed_target_pos
            3 * target_history_len +    # target_history (positions)
            3 * target_history_len +    # target_history (velocities)
            1 +                     # delay_magnitude (normalized)
            7 * action_history_len      # action_history
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        print(f"\nObservation space dimension: {obs_dim}")
        print(f"  - Joint state: 21 (pos + vel + gravity)")
        print(f"  - Delayed target: 3 (position)")
        print(f"  - Target history positions: {3 * target_history_len}")
        print(f"  - Target history velocities: {3 * target_history_len}")
        print(f"  - Delay magnitude: 1")
        print(f"  - Action history: {7 * action_history_len}")
        print()

        # ============================================================
        # Normalization Constants
        # ============================================================
        self.max_joint_vel = 2.0  # rad/s (conservative estimate)
        self.max_gravity_torque = np.array([40.0, 40.0, 30.0, 30.0, 10.0, 5.0, 5.0])

        # Workspace bounds (for normalization)
        self.workspace_center = np.array([0.4, 0.0, 0.6])
        self.workspace_size = np.array([0.4, 0.4, 0.3])
        self.max_cartesian_vel = 0.5  # m/s

    def _print_configuration(self) -> None:
        """Print environment configuration."""
        print(f"\n{'='*70}")
        print(f"TELEOPERATION ENVIRONMENT CONFIGURATION")
        print(f"{'='*70}")
        print(f"Delay Configuration: {self.delay_simulator.config_name}")
        print(f"  - Action delay: {self.delay_simulator.get_action_delay()} steps "
              f"({self.delay_simulator.get_action_delay() * 1000 / self.control_freq:.1f}ms)")

        # Get observation delay range
        obs_min = self.delay_simulator._obs_delay_min_steps
        obs_max = self.delay_simulator._obs_delay_max_steps
        print(f"  - Obs delay range: {obs_min}-{obs_max} steps "
              f"({obs_min * 1000 / self.control_freq:.1f}-"
              f"{obs_max * 1000 / self.control_freq:.1f}ms)")

        print(f"\nController Configuration:")
        controller_info = self.remote_robot.get_controller_info()
        print(f"  - Adaptive PD: delay_threshold={controller_info['gain_adaptation']['delay_threshold']:.3f}s")
        print(f"  - Min gain ratio: {controller_info['gain_adaptation']['min_gain_ratio']:.2f}")
        print(f"  - IK max joint change: {controller_info['ik_solver_config']['max_joint_change']:.3f} rad")

        print(f"\nTrajectory Configuration:")
        print(f"  - Type: {self.trajectory_type.value}")
        print(f"  - Randomization: {self.randomize_trajectory}")
        print(f"{'='*70}\n")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1

        # Reset trajectory generator
        leader_start_pos, leader_info = self.leader.reset(seed=seed, options=options)

        # Reset remote robot and controllers
        self.remote_robot.reset(
            initial_qpos=self.initial_qpos,
            reset_controllers=True
        )

        # Clear and initialize history buffers
        self.position_history.clear()
        self.velocity_history.clear()
        self.action_history.clear()
        self.joint_pos_history.clear()
        self.joint_vel_history.clear()
        self.target_position_history.clear()

        # Get initial robot state
        initial_remote_state = self.remote_robot.get_state()

        # Initialize position and velocity history
        max_history_needed = max(
            self.target_history_len,
            self.delay_simulator._obs_delay_max_steps + 5
        )

        for _ in range(max_history_needed):
            self.position_history.append(leader_start_pos.copy())
            self.velocity_history.append(np.zeros(3))
            self.target_position_history.append(leader_start_pos.copy())

        # Initialize joint state history
        for _ in range(self.joint_history_len):
            self.joint_pos_history.append(initial_remote_state["joint_pos"].copy())
            self.joint_vel_history.append(initial_remote_state["joint_vel"].copy())

        # Initialize action history with zeros
        max_action_history_needed = max(
            self.action_history_len,
            self.delay_simulator._action_delay_steps + 5
        )
        for _ in range(max_action_history_needed):
            self.action_history.append(np.zeros(self.n_joints))

        return self._get_observation(), self._get_info()

    def _get_delayed_position(self) -> np.ndarray:
        """Get delayed target position based on current observation delay."""
        if not self.position_history:
            return np.zeros(3)

        buffer_length = len(self.position_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(buffer_length)

        if delay_steps == 0:
            return self.position_history[-1].copy()

        delay_index = min(delay_steps, buffer_length - 1)
        return self.position_history[-delay_index - 1].copy()

    def _get_delayed_velocity(self) -> np.ndarray:
        """Get delayed target velocity based on current observation delay."""
        if not self.velocity_history:
            return np.zeros(3)

        buffer_length = len(self.velocity_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(buffer_length)

        if delay_steps == 0:
            return self.velocity_history[-1].copy()

        delay_index = min(delay_steps, buffer_length - 1)
        return self.velocity_history[-delay_index - 1].copy()

    def _get_delayed_action(self) -> np.ndarray:
        """Get delayed action based on current action delay."""
        if not self.action_history:
            return np.zeros(self.n_joints)

        buffer_length = len(self.action_history)
        delay_steps = self.delay_simulator.get_action_delay_steps()

        if delay_steps == 0:
            return self.action_history[-1].copy()

        if buffer_length <= delay_steps:
            return np.zeros(self.n_joints)

        delay_index = min(delay_steps, buffer_length - 1)
        return self.action_history[-delay_index - 1].copy()

    def _compute_current_delay(self) -> float:
        """Compute total round-trip delay in seconds."""
        obs_delay_steps = self.delay_simulator.get_observation_delay_steps(
            len(self.position_history)
        )
        action_delay_steps = self.delay_simulator.get_action_delay_steps()
        total_delay_steps = obs_delay_steps + action_delay_steps
        return total_delay_steps / self.control_freq

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.current_step += 1

        # Get new reference trajectory point
        leader_output = self.leader.step()

        if isinstance(leader_output, tuple):
            new_leader_position = leader_output[0]
            leader_info = leader_output[4] if len(leader_output) > 4 else {}
        else:
            new_leader_position = leader_output
            leader_info = {}

        target_velocity = leader_info.get('velocity', np.zeros(3))

        # Update history buffers
        self.position_history.append(new_leader_position.copy())
        self.velocity_history.append(target_velocity.copy())
        self.target_position_history.append(new_leader_position.copy())
        self.action_history.append(action.copy())

        # Apply delays
        delayed_position = self._get_delayed_position()
        delayed_velocity = self._get_delayed_velocity()
        delayed_action = self._get_delayed_action()

        current_delay = self._compute_current_delay()

        # Execute control step (convert action from [-1,1] to [-0.5,0.5])
        action_percentage = np.clip(action, -1.0, 1.0) * 0.5

        step_info = self.remote_robot.step(
            target_pos=delayed_position,
            normalized_action=action_percentage,
            current_delay=current_delay,
            target_vel_cartesian=delayed_velocity,
        )

        # Update joint state history
        remote_state = self.remote_robot.get_state()
        self.joint_pos_history.append(remote_state["joint_pos"].copy())
        self.joint_vel_history.append(remote_state["joint_vel"].copy())

        # Compute tracking error (real-time, not delayed)
        follower_tcp_pos = self.remote_robot.get_ee_position()
        real_time_error = np.linalg.norm(new_leader_position - follower_tcp_pos)

        # Calculate reward and check termination
        reward = self._calculate_reward(
            cartesian_error=real_time_error,
            action=action_percentage,
            remote_state=remote_state,
            step_info=step_info
        )

        terminated, term_penalty = self._check_termination(
            cartesian_error=real_time_error,
            remote_state=remote_state
        )
        reward += term_penalty

        truncated = self.current_step >= self.max_episode_steps

        # Debug logging
        if self.current_step % 200 == 0:
            print(f"Step {self.current_step}: "
                  f"Delay={current_delay*1000:.1f}ms, "
                  f"GainRatio={step_info['gain_ratio']:.2f}, "
                  f"Error={real_time_error*1000:.1f}mm, "
                  f"Reward={reward:.2f}")

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _normalize_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Normalize all observation components to approximately [-1, 1]."""
        # Joint positions
        joint_pos_norm = 2 * (obs_dict['joint_pos'] - self.joint_limits_lower) / \
                         (self.joint_limits_upper - self.joint_limits_lower) - 1

        # Joint velocities
        joint_vel_norm = np.clip(obs_dict['joint_vel'] / self.max_joint_vel, -1, 1)

        # Gravity torques
        gravity_norm = np.clip(obs_dict['gravity_torque'] / self.max_gravity_torque, -1, 1)

        # Delayed target position
        delayed_target_norm = (obs_dict['delayed_target'] - self.workspace_center) / \
                              self.workspace_size
        delayed_target_norm = np.clip(delayed_target_norm, -1, 1)

        # Target position history
        target_history_pos = obs_dict['target_history_positions'].reshape(-1, 3)
        target_history_pos_norm = (target_history_pos - self.workspace_center) / \
                                  self.workspace_size
        target_history_pos_norm = np.clip(target_history_pos_norm, -1, 1).flatten()

        # Target velocity history
        target_history_vel = obs_dict['target_history_velocities'].reshape(-1, 3)
        target_history_vel_norm = np.clip(target_history_vel / self.max_cartesian_vel, -1, 1).flatten()

        # Delay magnitude
        delay_norm = obs_dict['delay_magnitude']

        # Action history
        action_history_norm = obs_dict['action_history']

        # Concatenate all components
        observation = np.concatenate([
            joint_pos_norm,
            joint_vel_norm,
            gravity_norm,
            delayed_target_norm,
            target_history_pos_norm,
            target_history_vel_norm,
            delay_norm,
            action_history_norm
        ]).astype(np.float32)

        return observation

    def _get_observation(self) -> np.ndarray:
        """Get current normalized observation."""
        remote_state = self.remote_robot.get_state()

        # Current robot state
        joint_pos = remote_state['joint_pos']
        joint_vel = remote_state['joint_vel']
        gravity_torque = remote_state['gravity_torque']

        # Delayed target
        delayed_target = self._get_delayed_position()

        # Target position history
        target_pos_history = list(self.target_position_history)
        target_pos_history_flat = np.array(target_pos_history).flatten()

        # Target velocity history
        target_vel_list = list(self.velocity_history)[-self.target_history_len:]
        target_vel_history_flat = np.array(target_vel_list).flatten()

        # Delay magnitude
        obs_delay_steps = self.delay_simulator.get_observation_delay_steps(
            len(self.position_history)
        )
        delay_normalized = np.array([obs_delay_steps / 100.0])

        # Action history
        recent_actions = list(self.action_history)[-self.action_history_len:]
        action_history_flat = np.array(recent_actions).flatten()

        # Build observation dictionary
        obs_dict = {
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'gravity_torque': gravity_torque,
            'delayed_target': delayed_target,
            'target_history_positions': target_pos_history_flat,
            'target_history_velocities': target_vel_history_flat,
            'delay_magnitude': delay_normalized,
            'action_history': action_history_flat
        }

        return self._normalize_observation(obs_dict)

    def _calculate_reward(
        self,
        cartesian_error: float,
        action: np.ndarray,
        remote_state: Dict[str, np.ndarray],
        step_info: Dict[str, Any]
    ) -> float:
        """Calculate step reward for multiplicative residual learning."""
        cartesian_error = np.clip(cartesian_error, 0.0, 10.0)

        # Tracking reward
        tracking_reward = 10.0 * np.exp(-100.0 * cartesian_error**2)

        # Residual penalty
        residual_percentage = np.mean(np.abs(action))
        residual_penalty = -0.1 * residual_percentage

        # Smoothness penalty
        if len(self.action_history) >= 2:
            action_change = action - self.action_history[-1]
            smoothness_penalty = -0.5 * np.sum(action_change**2)
        else:
            smoothness_penalty = 0.0

        # Velocity penalty
        velocity_penalty = -0.01 * np.sum(remote_state['joint_vel']**2)

        # Joint limit proximity penalty
        joint_pos = remote_state['joint_pos']
        normalized_pos = (joint_pos - self.joint_limits_lower) / \
                         (self.joint_limits_upper - self.joint_limits_lower)
        limit_proximity = np.maximum(0, normalized_pos - 0.9) + \
                          np.maximum(0, 0.1 - normalized_pos)
        limit_penalty = -5.0 * np.sum(limit_proximity)

        # Precision bonus
        if cartesian_error < 0.01 and step_info['gain_ratio'] > 0.8:
            precision_bonus = 1.0
        else:
            precision_bonus = 0.0

        total_reward = (
            tracking_reward +
            residual_penalty +
            smoothness_penalty +
            velocity_penalty +
            limit_penalty +
            precision_bonus
        )

        return total_reward

    def _check_termination(
        self,
        cartesian_error: float,
        remote_state: Dict[str, np.ndarray]
    ) -> Tuple[bool, float]:
        """Check termination conditions."""
        termination_penalty = 0.0
        remote_pos = remote_state["joint_pos"]

        # Check joint limits
        at_lower_limit = np.any(
            remote_pos <= self.joint_limits_lower + self.joint_limit_margin
        )
        at_upper_limit = np.any(
            remote_pos >= self.joint_limits_upper - self.joint_limit_margin
        )
        at_limits = at_lower_limit or at_upper_limit

        # Check tracking error
        high_error = cartesian_error > self.max_cartesian_error

        terminated = at_limits or high_error

        if terminated:
            termination_penalty = -100.0

            if at_limits:
                violated_joints = np.where(
                    (remote_pos <= self.joint_limits_lower + self.joint_limit_margin) |
                    (remote_pos >= self.joint_limits_upper - self.joint_limit_margin)
                )[0]
                print(f"Episode {self.episode_count} Step {self.current_step} "
                      f"terminated: Joint limits exceeded at joints {violated_joints}")

            if high_error:
                print(f"Episode {self.episode_count} Step {self.current_step} "
                      f"terminated: High tracking error ({cartesian_error*1000:.1f}mm)")

        return terminated, termination_penalty

    def _get_info(self) -> Dict[str, Any]:
        """Get comprehensive episode information."""
        current_pos = self.position_history[-1] if self.position_history else np.zeros(3)
        delayed_pos = self._get_delayed_position()
        follower_pos = self.remote_robot.get_ee_position()

        real_time_error = np.linalg.norm(current_pos - follower_pos)
        delay_magnitude_spatial = np.linalg.norm(current_pos - delayed_pos)
        current_delay = self._compute_current_delay()

        debug_info = self.remote_robot.get_debug_info()
        remote_state = self.remote_robot.get_state()
        controller_gains = remote_state['controller_gains']

        info = {
            'real_time_cartesian_error': real_time_error,
            'real_time_cartesian_error_mm': real_time_error * 1000,
            'current_delay_seconds': current_delay,
            'current_delay_ms': current_delay * 1000,
            'delay_magnitude_spatial': delay_magnitude_spatial,
            'gain_ratio': controller_gains['gain_ratio'],
            'current_kp_mean': np.mean(controller_gains['kp']),
            'current_kd_mean': np.mean(controller_gains['kd']),
            'config_name': self.delay_simulator.config_name,
            'experiment_config': self.experiment_config.value,
            'trajectory_type': self.trajectory_type.value,
            'control_mode': 'adaptive_pd_inverse_dynamics_multiplicative',
        }

        if debug_info:
            info['tau_baseline_norm'] = np.linalg.norm(debug_info.get('tau_baseline', np.zeros(7)))
            info['tau_total_norm'] = np.linalg.norm(debug_info.get('tau_total', np.zeros(7)))
            info['tau_clipped_norm'] = np.linalg.norm(debug_info.get('tau_clipped', np.zeros(7)))

            if 'action_clipped' in debug_info:
                info['mean_correction_percentage'] = np.mean(np.abs(debug_info['action_clipped'])) * 100
                info['max_correction_percentage'] = np.max(np.abs(debug_info['action_clipped'])) * 100

            info['ik_success'] = debug_info.get('ik_success', False)
            info['ik_error'] = debug_info.get('ik_error', 0.0)
            info['joint_position_error'] = debug_info.get('position_error_joint', 0.0)
            info['joint_velocity_error'] = debug_info.get('velocity_error_joint', 0.0)
            info['limits_hit'] = debug_info.get('limits_hit', False)

        return info

    def render(self) -> None:
        """Render environment (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass
