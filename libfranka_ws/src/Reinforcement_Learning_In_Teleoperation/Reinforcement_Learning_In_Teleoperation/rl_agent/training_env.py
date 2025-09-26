"""
The main RL training environment with delay simulation.

We are going to train a full tau controller, using RL
"""

# RL library imports
import gymnasium as gym
from gymnasium import spaces

# Python standard libraries
import numpy as np
from collections import deque

# Custom modules
from local_robot_simulator import LocalRobotSimulator, TrajectoryType
from remote_robot_simulator import RemoteRobotSimulator
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator

class TeleoperationEnvWithDelay(gym.Env):
    metadata = {'render_modes': []}

    def __init__(
        self,
        model_path: str,
        experiment_config: int = 4,
        max_episode_steps: int = 500,
        control_freq: int = 500,
        max_cartesian_error: float = 1.0,
        ):
        
        # Training Configurations
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        self.current_step = 0
        
        self.max_cartesian_error = max_cartesian_error
        self.joint_limit_margin = 0.05 # Margin to avoid touching joint limits, safety margin = joint_limit - limit_margin

        # Robot configurations
        self.initial_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.torque_limit = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        self.tcp_offset = np.array([0.0, 0.0, 0.1034])
        self.n_joints = 7
        
        # Initialize delay simulator
        self.experiment_config = experiment_config
        self.delay_simulator = DelaySimulator(control_freq=control_freq, experiment_config=experiment_config)
        self.characteristic_torque = np.array([30.0, 30.0, 20.0, 20.0, 10.0, 5.0, 5.0])

        self.joint_history_len = 1
        self.action_history_len = 5
        
        # Get delay parameters from simulator
        max_obs_delay = max(self.delay_simulator.stochastic_obs_delay_max, 10)
        max_buffer_size = max(100, max_obs_delay + 20)
        self.position_history = deque(maxlen=max_buffer_size)
        self.action_history = deque(maxlen=max_buffer_size)
        self.joint_pos_history = deque(maxlen=self.joint_history_len)
        self.joint_vel_history = deque(maxlen=self.joint_history_len)

        # Initialize leader (local, perfect trajectory)
        self.leader = LocalRobotSimulator(
            control_freq=control_freq,
            trajectory_type=TrajectoryType.FIGURE_8,
            randomize_params=False                     # randomization
        )

        # Trajectory configuration
        self.trajectory_type = "figure_8"
        self.trajectory_scale = (0.1, 0.3)  # Fixed scale
        self.leader.set_trajectory_params(
            scale=np.array(self.trajectory_scale),  # Fixed [0.1, 0.3]
            frequency=0.1,                          # Fixed frequency
            center=np.array([0.4, 0.0, 0.6]),      # Fixed center
            initial_phase=0.0                       # Fixed phase
        )
        
        # Initialize remote robot (follower)
        self.remote_robot = RemoteRobotSimulator(
            model_path=model_path,
            control_freq=control_freq,
            torque_limits=self.torque_limit,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper
        )

        # Action space - normalized torques
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        
        # Observation space
        obs_dim = (
            (self.n_joints * self.joint_history_len) +  # joint positions
            (self.n_joints * self.joint_history_len) +  # joint velocities
            (3) +                                       # delayed positions
            (self.n_joints * self.action_history_len)  # action history
        )
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Episode tracking
        self.episode_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1
        
        # Rest Trajectory generator
        leader_start_pos, leader_info = self.leader.reset()
        
        # Reset remote robot
        self.remote_robot.reset(self.initial_qpos)

        # Clear histories buffer
        self.position_history.clear()
        self.action_history.clear()
        self.joint_pos_history.clear()
        self.joint_vel_history.clear()

        # Get initial remote robot state
        initial_remote_state = self.remote_robot.get_state()
        
        # Initialize position history
        max_history_needed = max(20, self.delay_simulator.stochastic_obs_delay_max)
        for _ in range(max_history_needed):
            self.position_history.append(leader_start_pos.copy())
            
        # Initialize joint histories
        for _ in range(self.joint_history_len):
            self.joint_pos_history.append(initial_remote_state["joint_pos"].copy())
            self.joint_vel_history.append(initial_remote_state["joint_vel"].copy())
            
        # Initialize action history
        max_action_history = max(self.action_history_len, self.delay_simulator.constant_action_delay + 1)
        for _ in range(max_action_history):
            self.action_history.append(np.random.uniform(-0.5, 0.5, self.n_joints))

        return self._get_observation(), self._get_info()

    def _get_delayed_position(self):
        """Get delayed position using DelaySimulator logic"""
        if not self.position_history:
            return np.zeros(3)
            
        buffer_length = len(self.position_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(buffer_length)
        
        if delay_steps == 0:
            return self.position_history[-1]
            
        delay_index = max(0, buffer_length - 1 - delay_steps)
        return self.position_history[delay_index]

    def _get_delayed_action(self):
        """Get delayed action using DelaySimulator logic"""
        if not self.action_history:
            return np.zeros(self.n_joints)
            
        buffer_length = len(self.action_history)
        delay_steps = self.delay_simulator.get_action_delay_steps(buffer_length)
        
        if delay_steps == 0:
            return self.action_history[-1]
            
        if buffer_length <= delay_steps:
            return np.zeros(self.n_joints)
            
        delay_index = max(0, buffer_length - 1 - delay_steps)
        return self.action_history[delay_index]

    def step(self, action: np.ndarray):
        self.current_step += 1
    
        # Get new leader position from trajectory
        leader_output = self.leader.step()
        new_leader_position = leader_output[0] if isinstance(leader_output, tuple) else leader_output
        self.position_history.append(new_leader_position.copy())
        self.action_history.append(action.copy())
        
        # Get delayed signals (for config=4, these will have no delay)
        delayed_position = self._get_delayed_position()
        delayed_action = self._get_delayed_action()
        target_tcp_pos = delayed_position
        
        # Send action to remote robot (FULL TAU CONTROL)
        self.remote_robot.step(
            target_pos=target_tcp_pos,
            normalized_action=delayed_action,
            characteristic_torque=self.characteristic_torque,
            action_delay_steps=self.delay_simulator.get_action_delay_steps(len(self.action_history))
        )
        
        # Get remote robot state after step
        remote_state = self.remote_robot.get_state()
        self.joint_pos_history.append(remote_state["joint_pos"].copy())
        self.joint_vel_history.append(remote_state["joint_vel"].copy())
        follower_tcp_pos = self.remote_robot.get_ee_position()
        real_time_error = np.linalg.norm(new_leader_position - follower_tcp_pos)
        
        # Calculate reward and termination
        reward = self._calculate_reward(real_time_error, action)
        terminated, term_penalty = self._check_termination(real_time_error)
        reward += term_penalty
        truncated = self.current_step >= self.max_episode_steps
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self) -> np.ndarray:
        """Get simplified observation for no-delay testing."""
        joint_pos_hist_flat = np.array(list(self.joint_pos_history)).flatten()
        joint_vel_hist_flat = np.array(list(self.joint_vel_history)).flatten()
        recent_actions = list(self.action_history)[-self.action_history_len:]
        action_history_flat = np.array(recent_actions).flatten()

        # Get current target position (no delay for config 4)
        current_target_pos = self._get_delayed_position()
        
        # Simple observation: joint state + target + action history
        observation = np.concatenate([
            joint_pos_hist_flat,
            joint_vel_hist_flat,
            current_target_pos,
            action_history_flat
        ]).astype(np.float32)

        return observation

    def _calculate_reward(self, cartesian_error: float, action: np.ndarray) -> float:
        """Calculate reward based on tracking error and action penalties"""
        cartesian_error = np.clip(cartesian_error, 0.0, 10.0)
        
        # Tracking reward
        if cartesian_error < 0.05:
            tracking_reward = 2.0 - 20 * cartesian_error
        elif cartesian_error < 0.1:
            tracking_reward = 1.0 - 10 * cartesian_error
        else:
            tracking_reward = np.exp(-2.0 * cartesian_error**2)
        
        # Action penalty (encourage smooth actions)
        action_penalty = -0.01 * np.sum(np.square(np.clip(action, -1.0, 1.0)))
        
        # Total reward
        total_reward = tracking_reward + action_penalty
        
        return total_reward

    def _check_termination(self, cartesian_error: float) -> tuple[bool, float]:
        """Check termination conditions."""
        termination_penalty = 0.0
        remote_state = self.remote_robot.get_state()
        remote_pos = remote_state["joint_pos"]
        
        # Check joint limits (with safety margin)
        at_limits = (
            np.any(remote_pos <= self.joint_limits_lower + self.joint_limit_margin) or
            np.any(remote_pos >= self.joint_limits_upper - self.joint_limit_margin)
        )
        
        # Check tracking error
        high_error = cartesian_error > self.max_cartesian_error
        
        terminated = at_limits or high_error
        if terminated:
            termination_penalty = -100.0
            if at_limits:
                print(f"Episode {self.episode_count} terminated: Joint limits exceeded")
            if high_error:
                print(f"Episode {self.episode_count} terminated: High error ({cartesian_error:.3f}m)")
        
        return terminated, termination_penalty
    
    def _get_info(self) -> dict:
        current_pos = self.position_history[-1] if self.position_history else np.zeros(3)
        delayed_pos = self._get_delayed_position()
        follower_pos = self.remote_robot.get_ee_position()
        real_time_error = np.linalg.norm(current_pos - follower_pos)
        delay_magnitude = np.linalg.norm(current_pos - delayed_pos)
        
        return {
            'real_time_cartesian_error': real_time_error,
            'delay_magnitude': delay_magnitude,
            'config_name': f"Config {self.experiment_config} (4=no_delay)",
            'experiment_config': self.experiment_config,
            'control_mode': 'full_tau',
            'trajectory_type': 'figure_8',
            'trajectory_scale': self.trajectory_scale
        }

    def close(self):
        pass