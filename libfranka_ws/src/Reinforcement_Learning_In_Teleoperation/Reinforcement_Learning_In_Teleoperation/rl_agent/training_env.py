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
        max_episode_steps: int = 1000,  # Changed from 500 to 1000
        control_freq: int = 500,
        max_cartesian_error: float = 1.0,
        ):
        
        # Training Configurations
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        self.current_step = 0
        
        self.max_cartesian_error = max_cartesian_error
        self.joint_limit_margin = 0.05

        # Robot configurations
        self.initial_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.torque_limit = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        self.n_joints = 7
        
        # Initialize delay simulator
        self.experiment_config = experiment_config
        self.delay_simulator = DelaySimulator(control_freq=control_freq, experiment_config=experiment_config)
        
        # REMOVED: self.characteristic_torque (computed adaptively now!)

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
            randomize_params=False
        )

        # Trajectory configuration
        self.trajectory_type = "figure_8"
        self.trajectory_scale = (0.1, 0.3)
        self.leader.set_trajectory_params(
            scale=np.array(self.trajectory_scale),
            frequency=0.1,
            center=np.array([0.4, 0.0, 0.6]),
            initial_phase=0.0
        )
        
        # Initialize remote robot (follower)
        self.remote_robot = RemoteRobotSimulator(
            model_path=model_path,
            control_freq=control_freq,
            torque_limits=self.torque_limit,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper
        )

        # Action space - normalized percentage corrections (±50%)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        
        # UPDATED: Observation space with gravity torque
        obs_dim = (
            (self.n_joints * self.joint_history_len) +  # joint positions
            (self.n_joints * self.joint_history_len) +  # joint velocities
            self.n_joints +                             # gravity torques (NEW!)
            3 +                                         # target position
            (self.n_joints * self.action_history_len)  # action history
        )
        
        # UPDATED: Bounded observation space
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Normalization constants
        self.max_joint_vel = 2.0  # rad/s
        self.max_gravity_torque = np.array([40.0, 40.0, 30.0, 30.0, 10.0, 5.0, 5.0])
        self.workspace_center = np.array([0.4, 0.0, 0.6])
        self.workspace_size = np.array([0.4, 0.4, 0.3])

        # Episode tracking
        self.episode_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1
        
        # Reset Trajectory generator
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
            
        # Initialize action history with zeros (not random!)
        max_action_history = max(self.action_history_len, self.delay_simulator.constant_action_delay + 1)
        for _ in range(max_action_history):
            self.action_history.append(np.zeros(self.n_joints))  # Changed from random

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
        
        # Get delayed signals
        delayed_position = self._get_delayed_position()
        delayed_action = self._get_delayed_action()
        
        # UPDATED: Call remote_robot.step() without characteristic_torque
        self.remote_robot.step(
            target_pos=delayed_position,
            normalized_action=delayed_action
        )
        
        # Get remote robot state after step
        remote_state = self.remote_robot.get_state()
        self.joint_pos_history.append(remote_state["joint_pos"].copy())
        self.joint_vel_history.append(remote_state["joint_vel"].copy())
        follower_tcp_pos = self.remote_robot.get_ee_position()
        real_time_error = np.linalg.norm(new_leader_position - follower_tcp_pos)
        
        # Calculate reward and termination
        reward = self._calculate_reward(real_time_error, action, remote_state)
        terminated, term_penalty = self._check_termination(real_time_error)
        reward += term_penalty
        truncated = self.current_step >= self.max_episode_steps
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _normalize_observation(self, obs_dict: dict) -> np.ndarray:
        """Normalize all observation components to [-1, 1]"""
        
        # Joint positions: normalize by limits
        joint_pos_norm = 2 * (obs_dict['joint_pos'] - self.joint_limits_lower) / \
                         (self.joint_limits_upper - self.joint_limits_lower) - 1
        
        # Joint velocities: clip and normalize
        joint_vel_norm = np.clip(obs_dict['joint_vel'] / self.max_joint_vel, -1, 1)
        
        # Gravity torques: normalize by maximum expected values
        gravity_norm = np.clip(obs_dict['gravity_torque'] / self.max_gravity_torque, -1, 1)
        
        # Target position: normalize by workspace
        target_pos_norm = (obs_dict['target_pos'] - self.workspace_center) / self.workspace_size
        target_pos_norm = np.clip(target_pos_norm, -1, 1)
        
        # Action history: already normalized
        action_history_norm = obs_dict['action_history']
        
        # Concatenate
        observation = np.concatenate([
            joint_pos_norm,
            joint_vel_norm,
            gravity_norm,
            target_pos_norm,
            action_history_norm
        ]).astype(np.float32)
        
        return observation

    def _get_observation(self) -> np.ndarray:
        """Get normalized observation with gravity torque"""
        remote_state = self.remote_robot.get_state()
        
        # Current state
        joint_pos = remote_state['joint_pos']
        joint_vel = remote_state['joint_vel']
        gravity_torque = remote_state['gravity_torque']  # NEW!
        
        # Target position
        current_target_pos = self._get_delayed_position()
        
        # Action history
        recent_actions = list(self.action_history)[-self.action_history_len:]
        action_history_flat = np.array(recent_actions).flatten()
        
        # Build observation dictionary
        obs_dict = {
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'gravity_torque': gravity_torque,  # NEW!
            'target_pos': current_target_pos,
            'action_history': action_history_flat
        }
        
        return self._normalize_observation(obs_dict)

    def _calculate_reward(self, cartesian_error: float, action: np.ndarray, remote_state: dict) -> float:
        """
        UPDATED: Reward for multiplicative residual learning
        """
        cartesian_error = np.clip(cartesian_error, 0.0, 10.0)
        
        # Tracking reward (smooth exponential)
        tracking_reward = 10.0 * np.exp(-100.0 * cartesian_error**2)
        
        # Residual percentage penalty
        # Since action represents percentage correction, penalize large percentages
        residual_percentage = np.mean(np.abs(action))
        residual_penalty = -0.1 * residual_percentage
        
        # Action smoothness penalty
        if len(self.action_history) >= 2:
            action_change = action - self.action_history[-1]
            smoothness_penalty = -0.5 * np.sum(action_change**2)
        else:
            smoothness_penalty = 0.0
        
        # Velocity penalty
        velocity_penalty = -0.01 * np.sum(remote_state['joint_vel']**2)
        
        # Total reward
        total_reward = (
            tracking_reward +
            residual_penalty +
            smoothness_penalty +
            velocity_penalty
        )
        
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
        
        # Get debug info from robot
        debug_info = self.remote_robot.get_debug_info()
        
        info = {
            'real_time_cartesian_error': real_time_error,
            'delay_magnitude': delay_magnitude,
            'config_name': f"Config {self.experiment_config}",
            'experiment_config': self.experiment_config,
            'control_mode': 'inverse_dynamics_multiplicative',
            'trajectory_type': self.trajectory_type,
            'trajectory_scale': self.trajectory_scale
        }
        
        # Add torque decomposition info if available
        if debug_info:
            info['tau_baseline_norm'] = np.linalg.norm(debug_info.get('tau_baseline', 0))
            info['tau_total_norm'] = np.linalg.norm(debug_info.get('tau_total', 0))
            if 'action_clipped' in debug_info:
                info['mean_correction_percentage'] = np.mean(np.abs(debug_info['action_clipped'])) * 100
        
        return info

    def close(self):
        pass