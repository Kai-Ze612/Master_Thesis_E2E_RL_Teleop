# RL library imports
import gymnasium as gym
from gymnasium import spaces

# Python standard libraries
import numpy as np
from collections import deque
import os

# Custom modules
from local_robot_simulator import LocalRobotSimulator
from remote_robot_simulator import RemoteRobotSimulator
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator


class TeleoperationEnvWithDelay(gym.Env):
    metadata = {'render_modes': []}

    def __init__(
        self,
        model_path: str,
        experiment_config: int = 1,
        max_episode_steps: int = 500,
        control_freq: int = 500,
        robot_config: dict = None,
        max_cartesian_error: float = 0.3,
        prediction_method: str = "linear_extrapolation"
        ):
        
        self.episode_count = 0

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        if experiment_config not in [1, 2, 3]:
            raise ValueError(f"Invalid experiment_config: {experiment_config}. Must be 1, 2, or 3.")

        if not (hasattr(LocalRobotSimulator, 'reset') and hasattr(LocalRobotSimulator, 'step')):
            raise ImportError("LocalRobotSimulator must implement reset and step methods.")
        if not (hasattr(RemoteRobotSimulator, 'reset') and hasattr(RemoteRobotSimulator, 'step')):
            raise ImportError("RemoteRobotSimulator must implement reset and step methods.")

        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        self.current_step = 0
        self.n_joints = 7
        self.max_cartesian_error = max_cartesian_error
        self.joint_limit_margin = 0.05
        self.prediction_method = prediction_method

        robot_config = {
            "initial_qpos": np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
            "joint_limits_lower": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            "joint_limits_upper": np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
            "torque_limit": np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
            "default_kp": np.array([10, 10, 10, 10, 10, 10, 10]),
            "default_kd": np.array([2, 2, 2, 2, 2, 2, 2]),
            "tcp_offset": np.array([0.0, 0.0, 0.1034])
        }
        
        self.initial_qpos = robot_config["initial_qpos"]
        self.joint_limits_lower = robot_config["joint_limits_lower"]
        self.joint_limits_upper = robot_config["joint_limits_upper"]
        self.torque_limit = robot_config["torque_limit"]
        self.default_kp = robot_config["default_kp"]
        self.default_kd = robot_config["default_kd"]
        self.tcp_offset = robot_config["tcp_offset"]

        print(f"Using prediction method: {self.prediction_method}")

        # Initialize delay simulator
        self.delay_simulator = DelaySimulator(control_freq=control_freq, experiment_config=experiment_config)

        if experiment_config == 4:  # No delay baseline
            self.characteristic_torque = self.torque_limit * 0
        else:
            self.characteristic_torque = np.array([30.0, 30.0, 20.0, 20.0, 10.0, 0.0, 0.0])

        self.experiment_config = experiment_config
        self.joint_history_len = 1
        self.action_history_len = 5
        
        # Get delay parameters from simulator
        max_obs_delay = max(self.delay_simulator.stochastic_obs_delay_max, 10)
        max_buffer_size = max(100, max_obs_delay + 20)
        self.position_history = deque(maxlen=max_buffer_size)
        self.action_history = deque(maxlen=max_buffer_size)
        self.joint_pos_history = deque(maxlen=self.joint_history_len)
        self.joint_vel_history = deque(maxlen=self.joint_history_len)

        self.leader = LocalRobotSimulator(control_freq=control_freq)
        self.remote_robot = RemoteRobotSimulator(
            model_path=model_path,
            control_freq=control_freq,
            default_kp=self.default_kp,
            default_kd=self.default_kd,
            torque_limits=self.torque_limit,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        
        # Observation space based on prediction method
        if self.prediction_method == "delayed_obs":
            n_delayed_positions = 5  # Include last 5 delayed positions
            obs_dim = (
                (self.n_joints * self.joint_history_len) +  # joint positions
                (self.n_joints * self.joint_history_len) +  # joint velocities
                (3 * n_delayed_positions) +                # multiple delayed positions
                (self.n_joints * self.action_history_len)  # action history
            )
        else:
            obs_dim = (
                (self.n_joints * self.joint_history_len) +  # joint positions
                (self.n_joints * self.joint_history_len) +  # joint velocities
                3 +                                         # predicted position
                3 +                                         # velocity estimate
                (self.n_joints * self.action_history_len)  # action history
            )
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1
        
        leader_start_pos, _ = self.leader.reset()
        self.remote_robot.reset(self.initial_qpos)
        self.position_history.clear()
        self.action_history.clear()
        self.joint_pos_history.clear()
        self.joint_vel_history.clear()

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

    def _predict_current_position(self, delay_steps: int):
        """Predict current leader position using simple physics-based methods"""
        if len(self.position_history) < 3:
            return self.position_history[-1] if self.position_history else np.zeros(3)
        
        if self.prediction_method == "delayed_obs":
            # Return the delayed observation directly
            delay_index = max(0, len(self.position_history) - 1 - delay_steps)
            return self.position_history[delay_index]
        
        elif self.prediction_method == "linear_extrapolation":
            # Simple linear extrapolation
            recent_positions = list(self.position_history)[-2:]
            dt = 1.0 / self.control_freq
            
            # Calculate velocity
            velocity = (recent_positions[-1] - recent_positions[-2]) / dt
            
            # Extrapolate forward
            predicted_pos = recent_positions[-1] + velocity * (delay_steps * dt)
            return predicted_pos
        
        elif self.prediction_method == "velocity_based":
            # Average velocity over multiple steps for stability
            if len(self.position_history) >= 5:
                recent_positions = np.array(list(self.position_history)[-5:])
                dt = 1.0 / self.control_freq
                
                # Calculate velocities and average them
                velocities = np.diff(recent_positions, axis=0) / dt
                avg_velocity = np.mean(velocities, axis=0)
                
                # Extrapolate with damping
                damping_factor = 0.8
                predicted_pos = recent_positions[-1] + avg_velocity * (delay_steps * dt) * damping_factor
                return predicted_pos
            else:
                # Fallback to linear extrapolation
                return self._predict_current_position(delay_steps)
        
        # Default fallback
        return self.position_history[-1]

    def step(self, action: np.ndarray):
        self.current_step += 1
    
        leader_output = self.leader.step()
        new_leader_position = leader_output[0] if isinstance(leader_output, tuple) else leader_output
        self.position_history.append(new_leader_position.copy())
        self.action_history.append(action.copy())
        delayed_position = self._get_delayed_position()
        delayed_action = self._get_delayed_action()
        target_tcp_pos = delayed_position
        
        self.remote_robot.step(
            target_pos=target_tcp_pos,
            normalized_action=delayed_action,
            characteristic_torque=self.characteristic_torque,
            action_delay_steps=self.delay_simulator.get_action_delay_steps(len(self.action_history))
        )
        
        remote_state = self.remote_robot.get_state()
        self.joint_pos_history.append(remote_state["joint_pos"].copy())
        self.joint_vel_history.append(remote_state["joint_vel"].copy())
        follower_tcp_pos = self.remote_robot.get_ee_position()
        real_time_error = np.linalg.norm(new_leader_position - follower_tcp_pos)
        
        reward = self._calculate_reward(real_time_error, action)
        
        terminated, term_penalty = self._check_termination(real_time_error)
        reward += term_penalty
        
        truncated = self.current_step >= self.max_episode_steps
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self) -> np.ndarray:
        joint_pos_hist_flat = np.array(list(self.joint_pos_history)).flatten()
        joint_vel_hist_flat = np.array(list(self.joint_vel_history)).flatten()
        recent_actions = list(self.action_history)[-self.action_history_len:]
        action_history_flat = np.array(recent_actions).flatten()

        # Get current delay
        buffer_length = len(self.position_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(buffer_length)

        if self.prediction_method == "delayed_obs":
            # Include multiple delayed positions
            delayed_positions = []
            for i in range(5):  # Last 5 delayed positions
                delay_idx = max(0, buffer_length - 1 - delay_steps - i)
                if delay_idx < len(self.position_history):
                    delayed_positions.extend(self.position_history[delay_idx])
                else:
                    delayed_positions.extend([0.0, 0.0, 0.0])
            
            observation = np.concatenate([
                joint_pos_hist_flat,
                joint_vel_hist_flat,
                np.array(delayed_positions),
                action_history_flat
            ]).astype(np.float32)
            
        else:  # linear_extrapolation or velocity_based
            # Predict current position
            predicted_pos = self._predict_current_position(delay_steps)
            
            # Calculate velocity estimate
            if len(self.position_history) >= 2:
                dt = 1.0 / self.control_freq
                velocity_estimate = (self.position_history[-1] - self.position_history[-2]) / dt
            else:
                velocity_estimate = np.zeros(3)
            
            observation = np.concatenate([
                joint_pos_hist_flat,
                joint_vel_hist_flat,
                predicted_pos,
                velocity_estimate,
                action_history_flat
            ]).astype(np.float32)

        return observation

    def _calculate_reward(self, cartesian_error: float, action: np.ndarray) -> float:
        """Calculate reward based on tracking error and action penalties"""
        cartesian_error = np.clip(cartesian_error, 0.0, 10.0)
        alpha = 2.0
        
        # Calculate tracking reward
        if cartesian_error < 0.1:
            tracking_reward = 1.0 - 10 * cartesian_error
        else:
            tracking_reward = np.exp(-alpha * cartesian_error**2)
        
        # Calculate action penalty
        action_penalty = -0.1 * np.sum(np.square(np.clip(action, -1.0, 1.0)))
        
        # Total reward
        total_reward = tracking_reward + action_penalty
        
        return total_reward

    def _check_termination(self, cartesian_error: float) -> tuple[bool, float]:
        termination_penalty = 0.0
        remote_state = self.remote_robot.get_state()
        remote_pos = remote_state["joint_pos"]
        at_limits = (
            np.any(remote_pos <= self.joint_limits_lower + self.joint_limit_margin) or
            np.any(remote_pos >= self.joint_limits_upper - self.joint_limit_margin)
        )
        high_error = cartesian_error > self.max_cartesian_error
        terminated = at_limits or high_error
        if terminated:
            termination_penalty = -500.0
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
            'config_name': f"{self.delay_simulator.delay_config_name if hasattr(self.delay_simulator, 'delay_config_name') else f'Config {self.experiment_config}'} ({self.prediction_method})",
            'prediction_method': self.prediction_method
        }

    def close(self):
        pass