"""
The remote robot serves as a follower device.

This node receives Cartesian position commands from the local robot and executes them on the real robot arm.
"""

# ROS2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory
from collections import deque

# MuJoCo imports
import mujoco

# Python imports
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# RL imports
import torch
from stable_baselines3 import SAC

# Custom IK imports
from Hierarchical_RL_Teleoperation.controllers.inverse_kinematics import InverseKinematicsSolver
from Hierarchical_RL_Teleoperation.controllers.pd_controller import PDController
from Hierarchical_RL_Teleoperation.utils.delay_simulator import DelaySimulator

class RemoteRobot(Node):
    def __init__(self):
        super().__init__('remote_robot_follower')
        
        self._init_parameters()
        self._init_mujoco()
        self._init_controllers()
        self._init_compensation_model()
        self._init_ros_interfaces()
        self._init_delay_simulation()

        self.get_logger().info("Remote Robot Node with RL compensation (no LSTM) initialized")

    def _init_parameters(self):
        # Robot parameters
        self.num_joints = 7
        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        # Control frequencies
        self.publish_freq = 100  # Hz for messages publishing
        self.control_freq = 500  # Hz for control loop
        
        # Joint limits
        self.joint_limits_lower = np.array([
            -2.8973, -1.7628, -2.8973, -3.0718,
            -2.8973, -0.0175, -2.8973
        ])
        
        self.joint_limits_upper = np.array([
            2.8973, 1.7628, 2.8973, -0.0698,
            2.8973, 3.7525, 2.8973
        ]) # radians

        # Torque Limit
        self.torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])  # Nm

        # TCP offset
        self.ee_offset = np.array([0.0, 0.0, 0.1034])
        
        # Robot states
        self.real_joint_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.real_joint_velocities = np.zeros(self.num_joints)
        self.initial_qpos = self.real_joint_positions.copy()

        # PD gains - Match your training environment
        self.kp = np.array([80.0, 70.0, 60.0, 50.0, 30.0, 20.0, 20.0])
        self.kd = 2 * np.sqrt(self.kp)  # For critical damping

        # Control targets
        self.current_q_target = self.real_joint_positions.copy()
        self.last_q_target = None
        self.last_time = None
        self.velocity_filter_alpha = 0.1
        self.max_joint_velocity = 0.75
        self.smoothed_q_dot_target = np.zeros(self.num_joints)

        # Joint space interpolation parameters
        self.max_q_step = 0.003

        # Connection monitoring
        self.robot_connected = False
        
        # Debug counters
        self.debug_counter = 0
        self.debug_freq = 25

        # RL Parameters - Updated for LSTM-free environment
        self.joint_history_len = 1
        self.action_history_len = 5
        self.characteristic_torque = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 0.0, 0.0])
        
        # Prediction method - CHOOSE ONE: "linear_extrapolation", "velocity_based", "delayed_obs"
        self.prediction_method = "linear_extrapolation"

        # Model Path - Update this to your LSTM-free trained model
        self.rl_model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/rl_agent/rl_training_output/saved_model_3/models/best_model.zip"
        # History buffers for RL observation
        self.joint_pos_history = deque(maxlen=self.joint_history_len)
        self.joint_vel_history = deque(maxlen=self.joint_history_len)
        self.action_history_extended = deque(maxlen=5)
        
        # Initialize histories
        initial_joint_pos = self.real_joint_positions.copy()
        initial_joint_vel = np.zeros(self.num_joints)
        
        self.joint_pos_history = deque([initial_joint_pos] * self.joint_history_len, maxlen=self.joint_history_len)
        self.joint_vel_history = deque([initial_joint_vel] * self.joint_history_len, maxlen=self.joint_history_len)
        
        # Initialize action history with zeros
        for _ in range(5):
            self.action_history_extended.append(np.zeros(self.num_joints))

    def _init_mujoco(self):
        franka_description_path = get_package_share_directory('franka_description')
        model_path = os.path.join(franka_description_path, 'mujoco', 'franka', 'scene.xml')
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"MuJoCo model not found at {model_path}!")
            raise FileNotFoundError("Simulation model not found")
        
        self.kinematics_model = mujoco.MjModel.from_xml_path(model_path)
        self.kinematics_data = mujoco.MjData(self.kinematics_model)
        self.kinematics_model.opt.gravity[2] = -9.81
        self.get_logger().info("MuJoCo model loaded for kinematics.")
    
    def _init_controllers(self):
        """Initialize IK solver and PD controller."""
        # Initialize IK solver
        self.ik_solver = InverseKinematicsSolver(
            self.kinematics_model,
            self.joint_limits_lower,
            self.joint_limits_upper
        )
        
        # Initialize PD controller
        self.pd_controller = PDController(
            kp=self.kp,
            kd=self.kd,
            torque_limits=self.torque_limits,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper
        )
        
        self.get_logger().info("Controllers initialized")

    def _init_compensation_model(self):
        """Load the RL compensation model (no LSTM needed)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device '{self.device}' for RL model.")

        self.rl_model = self._load_rl_agent()
        self.get_logger().info(f"Using prediction method: {self.prediction_method}")
        
    def _load_rl_agent(self):
        try:
            if not os.path.exists(self.rl_model_path):
                self.get_logger().error(f"RL model not found at {self.rl_model_path}")
                self.get_logger().error("Please update the path to your trained LSTM-free model")
                raise FileNotFoundError(f"RL model not found: {self.rl_model_path}")
                
            model = SAC.load(self.rl_model_path, device=self.device)
            self.get_logger().info("Successfully loaded SAC RL agent (LSTM-free).")
            return model
        except Exception as e:
            self.get_logger().error(f"Failed to load RL agent model: {e}")
            raise

    def _init_ros_interfaces(self):
        """Initialize ROS interfaces."""
        # Subscriber for Cartesian pose commands from local robot
        self.cartesian_pose_sub = self.create_subscription(
            PoseStamped, '/local_robot/leader_pose',
            self.cartesian_pose_callback, 10
        )

        # Subscriber to joint states
        self.joint_state_sub = self.create_subscription(
            JointState, '/franka/joint_states',
            self.joint_state_callback, 10
        )
        
        # Publisher: tau for real robot
        self.torque_cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_tau/torques_desired', 10
        )
        
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, 'remote_robot/end_effector_pose', 10
        )
        
        # Control loop timer
        self.control_timer = self.create_timer(1.0/self.control_freq, self.control_loop)
        
        # Publisher timer
        self.state_publish_timer = self.create_timer(1.0/self.publish_freq, self.publish_end_effector_state)
    
    def _init_delay_simulation(self):
        """Initialize delay simulation using your existing DelaySimulator class."""
        self.experiment_config = 3
                
        # Initialize your DelaySimulator
        self.delay_simulator = DelaySimulator(
            control_freq=self.control_freq,
            experiment_config=self.experiment_config
        )
        
        # Position history buffer for observation delay
        max_delay_steps = 1000
        self.position_history = deque(maxlen=max_delay_steps)

        self.get_logger().info(f"DelaySimulator initialized with config {self.experiment_config}")

    def _get_current_ee_pose_from_mujoco(self):
        """Computes the end-effector pose using the existing MuJoCo model."""
        if not np.all(np.isfinite(self.real_joint_positions)):
            self.get_logger().warn("Invalid joint positions for FK computation.")
            return None, None
        
        self.kinematics_data.qpos[:self.num_joints] = self.real_joint_positions
        mujoco.mj_forward(self.kinematics_model, self.kinematics_data)
        
        ee_body_name = 'panda_hand'
        position = self.kinematics_data.body(ee_body_name).xpos.copy()
        orientation_matrix = self.kinematics_data.body(ee_body_name).xmat.copy().reshape(3, 3)
        
        tcp_offset_rotated = orientation_matrix @ self.ee_offset
        final_position = position + tcp_offset_rotated
        
        return final_position, orientation_matrix
        
    def cartesian_pose_callback(self, msg):
        """Stores incoming leader positions in a history buffer."""
        target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.position_history.append(target_position)

    def joint_state_callback(self, msg):
        """Update real robot state from hardware and set connected flag on first message."""
        try:
            name_to_idx_map = {name: i for i, name in enumerate(msg.name)}
            
            for i, name in enumerate(self.joint_names):
                if name in name_to_idx_map:
                    idx = name_to_idx_map[name]
                    self.real_joint_positions[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self.real_joint_velocities[i] = msg.velocity[idx]
                                
            if not self.robot_connected:
                self.current_q_target = self.real_joint_positions.copy()
                self.robot_connected = True
                self.get_logger().info("First joint state received. Robot is connected and controller is active!")
                
        except Exception as e:
            self.get_logger().error(f"Error processing joint states: {e}")

    def _get_delayed_position(self):
        """Gets a delayed position from the history buffer using the DelaySimulator."""
        if not self.position_history:
            return None
        
        delay_steps = self.delay_simulator.get_observation_delay_steps(len(self.position_history))
        
        if delay_steps >= len(self.position_history):
            return self.position_history[0]
            
        delay_index = len(self.position_history) - 1 - delay_steps
        return self.position_history[delay_index]

    def _predict_current_position(self, delay_steps: int):
        """Predict current leader position using physics-based methods (no LSTM)."""
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
                return self._predict_current_position_linear(delay_steps)
        
        # Default fallback
        return self.position_history[-1]

    def _get_rl_observation(self) -> np.ndarray:
        """Constructs the observation vector matching the LSTM-free training environment exactly."""
        
        # 1. Joint positions and velocities
        joint_pos_hist_flat = np.array(list(self.joint_pos_history)).flatten()  # 7 elements
        joint_vel_hist_flat = np.array(list(self.joint_vel_history)).flatten()  # 7 elements
        
        # 2. Get current delay
        delay_steps = self.delay_simulator.get_observation_delay_steps(len(self.position_history))
        
        # 3. Physics-based prediction instead of LSTM
        if self.prediction_method == "delayed_obs":
            # Include multiple delayed positions
            delayed_positions = []
            for i in range(5):  # Last 5 delayed positions
                delay_idx = max(0, len(self.position_history) - 1 - delay_steps - i)
                if delay_idx < len(self.position_history):
                    delayed_positions.extend(self.position_history[delay_idx])
                else:
                    delayed_positions.extend([0.0, 0.0, 0.0])
            
            # 4. Action history
            recent_actions = list(self.action_history_extended)[-5:]
            action_history_flat = np.array(recent_actions).flatten()  # 35 elements
            
            observation = np.concatenate([
                joint_pos_hist_flat,        # 7 elements
                joint_vel_hist_flat,        # 7 elements  
                np.array(delayed_positions), # 15 elements (5 positions × 3)
                action_history_flat         # 35 elements
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
            
            # 4. Action history
            recent_actions = list(self.action_history_extended)[-5:]
            action_history_flat = np.array(recent_actions).flatten()  # 35 elements
            
            observation = np.concatenate([
                joint_pos_hist_flat,        # 7 elements
                joint_vel_hist_flat,        # 7 elements  
                predicted_pos,              # 3 elements
                velocity_estimate,          # 3 elements
                action_history_flat         # 35 elements
            ]).astype(np.float32)
        
        return observation

    def control_loop(self):
        """
        Enhanced control loop with RL compensation using physics-based prediction.
        """
        if not self.robot_connected or not self.position_history:
            if self.robot_connected:
                self.current_q_target = self.real_joint_positions.copy()
            return

        self.debug_counter += 1

        # 1. GET DELAYED CARTESIAN TARGET
        delayed_position = self._get_delayed_position()
        if delayed_position is None:
            return

        # 2. COMPUTE GOAL JOINT STATE VIA IK
        q_goal, ik_error = self.ik_solver.solver(
            delayed_position,
            self.real_joint_positions,  # Use current state as seed
            ee_body_name='panda_hand'
        )
        if q_goal is None:
            if self.debug_counter % self.debug_freq == 0:
                self.get_logger().warn(f"IK failed. Holding last valid target. Error: {ik_error:.4f}")
            q_goal = self.current_q_target.copy()

        # 3. JOINT SPACE INTERPOLATION
        q_error = q_goal - self.current_q_target
        error_norm = np.linalg.norm(q_error)
        
        if error_norm > self.max_q_step:
            step = q_error / error_norm * self.max_q_step
            self.current_q_target += step
        elif error_norm > 1e-6:
            self.current_q_target = q_goal.copy()
        
        q_target = self.current_q_target

        # 4. CALCULATE SMOOTH TARGET VELOCITY
        current_time = self.get_clock().now().nanoseconds / 1e9
        noisy_q_dot_target = np.zeros(self.num_joints)
        if self.last_q_target is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 1e-6:
                noisy_q_dot_target = (q_target - self.last_q_target) / dt
        
        noisy_q_dot_target = np.clip(noisy_q_dot_target, -self.max_joint_velocity, self.max_joint_velocity)
        self.smoothed_q_dot_target = (1 - self.velocity_filter_alpha) * self.smoothed_q_dot_target + \
                                     self.velocity_filter_alpha * noisy_q_dot_target

        self.last_q_target = q_target.copy()
        self.last_time = current_time

        # 5. UPDATE RL OBSERVATION HISTORIES
        self.joint_pos_history.append(self.real_joint_positions.copy())
        self.joint_vel_history.append(self.real_joint_velocities.copy())

        # 6. GET RL COMPENSATION (using physics-based observation)
        observation = self._get_rl_observation()
        rl_action, _ = self.rl_model.predict(observation, deterministic=True)
        self.action_history_extended.append(rl_action.copy())
        
        # Convert RL action to compensation torque
        compensation_torque = rl_action * self.characteristic_torque

        # 7. PREDICT FUTURE ROBOT STATE
        action_delay_steps = self.delay_simulator.get_action_delay_steps(len(self.position_history))
        action_delay_seconds = action_delay_steps / self.control_freq
        predicted_joint_positions = self.real_joint_positions + self.real_joint_velocities * action_delay_seconds
        predicted_joint_velocities = self.real_joint_velocities

        # 8. COMPUTE PD TORQUES 
        pd_torques = self.pd_controller.compute_desired_acceleration(
            target_positions=q_target,
            target_velocities=self.smoothed_q_dot_target,
            current_positions=predicted_joint_positions,
            current_velocities=predicted_joint_velocities
        )
        
        # COMBINE PD + RL COMPENSATION
        final_torques = pd_torques + compensation_torque * 6
                
        # PUBLISH ENHANCED TORQUES
        torques_clipped = np.clip(final_torques, -self.torque_limits, self.torque_limits)
        cmd_msg = Float64MultiArray()
        cmd_msg.data = torques_clipped.tolist()
        self.torque_cmd_pub.publish(cmd_msg)

        if self.debug_counter % self.debug_freq == 0:
            obs_delay_val = self.delay_simulator.get_observation_delay_steps(len(self.position_history))
            self.get_logger().info(f"Prediction Method: {self.prediction_method}")
            self.get_logger().info(f"Delay Steps -> Obs: {obs_delay_val}, Action: {action_delay_steps}")
            self.get_logger().info(f"PD Torques Norm: {np.linalg.norm(pd_torques):.2f}")
            self.get_logger().info(f"RL Compensation Norm: {np.linalg.norm(compensation_torque):.2f}")
            self.get_logger().info(f"Published Torques Norm: {np.linalg.norm(torques_clipped):.2f}")

    def publish_end_effector_state(self):
        """Publishes the robot's current end-effector pose using the MuJoCo model."""
        if not self.robot_connected:
            return

        try:
            position, rotation_matrix = self._get_current_ee_pose_from_mujoco()
            
            if position is None:
                return
                
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "panda_link0"
            
            msg.pose.position.x = position[0]
            msg.pose.position.y = position[1]
            msg.pose.position.z = position[2]
            
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()
            
            msg.pose.orientation.x = quat[0]
            msg.pose.orientation.y = quat[1]
            msg.pose.orientation.z = quat[2]
            msg.pose.orientation.w = quat[3]
            
            self.ee_pose_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Error in publish_end_effector_state: {e}')

def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = RemoteRobot()
        node.get_logger().info("RL-enhanced remote robot node with physics-based prediction started.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down RL-enhanced remote robot follower...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()