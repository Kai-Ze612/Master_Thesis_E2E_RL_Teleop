"""
Remote Robot Controller with IK-Visualizer Synchronization

This version mirrors the real robot approach: synchronizing the IK solver 
joint positions with the MuJoCo visualizer, just like the real robot script 
synchronizes IK solver with physical robot hardware.
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

# AI Model imports
import torch
import torch.nn as nn
from stable_baselines3 import SAC

# Custom Controller imports
from Hierarchical_RL_Teleoperation.controllers.inverse_kinematics import InverseKinematicsSolver
from Hierarchical_RL_Teleoperation.controllers.pd_controller import PDController
from Hierarchical_RL_Teleoperation.utils.delay_simulator import DelaySimulator

class PositionLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class RemoteRobotController(Node):
    def __init__(self):
        super().__init__('remote_robot_follower')
        
        self._init_parameters()
        self._init_mujoco()
        self._init_controllers()
        self._init_compensation_model()
        self._init_ros_interfaces()
        self._init_delay_simulation()

        self.get_logger().info("Remote Robot Controller with IK-Visualizer Sync initialized")

    def _init_parameters(self):
        # Robot parameters
        self.num_joints = 7
        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        # Control frequencies
        self.publish_freq = 100  # Hz for messages publishing
        self.control_freq = 200  # Hz for control loop
        
        # Joint limits (rad)
        self.joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # Torque limits (Nm)
        self.torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

        # TCP offset
        self.ee_offset = np.array([0.0, 0.0, 0.1034])
        
        # Robot states - Initialize to safe position like real robot script
        self.real_joint_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.real_joint_velocities = np.zeros(self.num_joints)

        # PD gains
        self.kp = np.array([80.0, 70.0, 60.0, 50.0, 30.0, 20.0, 20.0])
        self.kd = 2 * np.sqrt(self.kp)  # Critical damping
        
        # Control targets
        self.current_q_target = self.real_joint_positions.copy()
        self.last_q_target = None
        self.last_time = None
        self.velocity_filter_alpha = 0.1
        self.max_joint_velocity = 0.75
        self.smoothed_q_dot_target = np.zeros(self.num_joints)

        # Joint space interpolation parameters
        self.max_q_step = 0.01

        # Connection monitoring
        self.robot_connected = False
        
        # Debug counters
        self.debug_counter = 0
        self.debug_freq = 25

        # Model Paths
        self.rl_model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/rl_agent/rl_training_output/SAC_DelayConfig3_Freq500_20250913-130108/models/best_model.zip"
        self.lstm_model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/state_predictor/models/best_lstm_model.pth"
        
        # RL observation parameters - must match training
        self.joint_history_len = 3
        self.action_history_len = 3
        self.past_leader_points_in_obs = 3
        self.lstm_sequence_length = 10
        self.characteristic_torque = self.torque_limits * 0.2

        # History buffers for constructing the observation
        self.action_history = deque(maxlen=self.action_history_len)
        self.joint_pos_history = deque(maxlen=self.joint_history_len)
        self.joint_vel_history = deque(maxlen=self.joint_history_len)

    def diagnose_coordinate_mismatch(self):
        """Test if IK solver and MuJoCo simulator use the same coordinate system."""
        
        # Step 1: Get current joint positions from simulation
        current_q = self.real_joint_positions.copy()
        self.get_logger().info(f"Current joint positions: {current_q}")
        
        # Step 2: Compute forward kinematics with MuJoCo using current joints
        self.kinematics_data.qpos[:self.num_joints] = current_q
        mujoco.mj_forward(self.kinematics_model, self.kinematics_data)
        
        ee_id = self.kinematics_model.body('panda_hand').id
        mujoco_ee_pos = self.kinematics_data.xpos[ee_id].copy()
        mujoco_ee_rot = self.kinematics_data.xmat[ee_id].copy().reshape(3, 3)
        
        self.get_logger().info(f"MuJoCo FK result: {mujoco_ee_pos}")
        
        # Step 3: Use IK solver to find joints for the SAME end-effector position
        ik_result, ik_error = self.ik_solver.solver(
            mujoco_ee_pos, 
            current_q,  # Use same starting point
            'panda_hand'
        )
        
        if ik_result is not None:
            self.get_logger().info(f"IK result for same position: {ik_result}")
            self.get_logger().info(f"Joint difference: {np.abs(ik_result - current_q)}")
            self.get_logger().info(f"Max joint difference: {np.max(np.abs(ik_result - current_q))}")
            
            # Test: If coordinate systems match, joint differences should be minimal
            if np.max(np.abs(ik_result - current_q)) > 0.1:  # 0.1 rad = ~6 degrees
                self.get_logger().error("COORDINATE MISMATCH DETECTED!")
                self.get_logger().error("IK solver and MuJoCo use different coordinate systems")
                return False
            else:
                self.get_logger().info("Coordinate systems appear consistent")
                return True
        else:
            self.get_logger().error(f"IK failed for current position. Error: {ik_error}")
            return False
        
    def _init_mujoco(self):
        """Initialize MuJoCo models for both IK and visualization sync."""
        franka_description_path = get_package_share_directory('franka_description')
        model_path = os.path.join(franka_description_path, 'mujoco', 'franka', 'scene.xml')
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"MuJoCo model not found at {model_path}!")
            raise FileNotFoundError("MuJoCo model not found")
        
        # Primary model for kinematics (like real robot script)
        self.kinematics_model = mujoco.MjModel.from_xml_path(model_path)
        self.kinematics_data = mujoco.MjData(self.kinematics_model)
        self.kinematics_model.opt.gravity[2] = -9.81
        
        # SYNC FIX: Initialize history buffers first (like real robot script)
        for _ in range(self.joint_history_len):
            self.joint_pos_history.append(self.real_joint_positions.copy())
            self.joint_vel_history.append(self.real_joint_velocities.copy())
        for _ in range(self.action_history_len):
            self.action_history.append(np.zeros(self.num_joints))
        
        self.get_logger().info("MuJoCo model loaded for kinematics and visualization sync")
    
    def check_model_differences(self):
        """Compare key model parameters between IK solver and MuJoCo."""
        
        self.get_logger().info("=== MODEL COMPARISON ===")
        
        # Check joint limits
        self.get_logger().info(f"IK joint limits lower: {self.joint_limits_lower}")
        self.get_logger().info(f"IK joint limits upper: {self.joint_limits_upper}")
        
        # Check MuJoCo joint limits
        mujoco_limits_lower = self.kinematics_model.jnt_range[:self.num_joints, 0]
        mujoco_limits_upper = self.kinematics_model.jnt_range[:self.num_joints, 1]
        
        self.get_logger().info(f"MuJoCo limits lower: {mujoco_limits_lower}")
        self.get_logger().info(f"MuJoCo limits upper: {mujoco_limits_upper}")
        
        # Check joint names/order
        for i in range(self.num_joints):
            joint_name = mujoco.mj_id2name(self.kinematics_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            expected_name = self.joint_names[i]
            self.get_logger().info(f"Joint {i}: MuJoCo='{joint_name}' Expected='{expected_name}'")
            
            if joint_name != expected_name:
                self.get_logger().error(f"JOINT NAME MISMATCH at index {i}")
        
        # Check end-effector body exists
        try:
            ee_id = self.kinematics_model.body('panda_hand').id
            self.get_logger().info(f"End-effector 'panda_hand' found with ID: {ee_id}")
        except:
            self.get_logger().error("End-effector 'panda_hand' not found in MuJoCo model")
            
            # List available bodies
            self.get_logger().info("Available bodies:")
            for i in range(self.kinematics_model.nbody):
                body_name = mujoco.mj_id2name(self.kinematics_model, mujoco.mjtObj.mjOBJ_BODY, i)
                self.get_logger().info(f"  {i}: {body_name}")

    def align_coordinate_systems(self):
        """Attempt to align IK solver with MuJoCo coordinate system."""
        
        # Option 1: Check if models use different joint conventions
        # Some models use degrees vs radians, or different zero positions
        
        # Option 2: Add coordinate transform
        # If there's a consistent transform, apply it to IK results
        
        # Option 3: Verify TCP offset
        self.get_logger().info(f"Using TCP offset: {self.ee_offset}")
        
        # Test with zero offset
        test_pos, _ = self._get_current_ee_pose_from_mujoco()
        if test_pos is not None:
            self.get_logger().info(f"Current EE position with offset: {test_pos}")
            
            # Try without offset
            self.kinematics_data.qpos[:self.num_joints] = self.real_joint_positions
            mujoco.mj_forward(self.kinematics_model, self.kinematics_data)
            ee_id = self.kinematics_model.body('panda_hand').id
            pos_no_offset = self.kinematics_data.xpos[ee_id].copy()
            self.get_logger().info(f"EE position without offset: {pos_no_offset}")
    
    def _init_controllers(self):
        """Initialize IK solver and PD controller."""
        self.ik_solver = InverseKinematicsSolver(
            self.kinematics_model,
            self.joint_limits_lower,
            self.joint_limits_upper
        )
        
        self.pd_controller = PDController(
            kp=self.kp,
            kd=self.kd,
            torque_limits=self.torque_limits,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper
        )
        
        self.get_logger().info("IK solver and PD controller initialized")

    def _init_compensation_model(self):
        """Load the RL compensation model and LSTM state predictor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device '{self.device}' for AI models")

        self.state_predictor = self._load_lstm_predictor()
        self.rl_model = self._load_rl_agent()
        
    def _load_lstm_predictor(self):
        try:
            model = PositionLSTM(input_size=3, hidden_size=128, num_layers=2).to(self.device)
            model.load_state_dict(torch.load(self.lstm_model_path, map_location=self.device)['model_state_dict'])
            model.eval()
            self.get_logger().info("Successfully loaded LSTM state predictor")
            return model
        except Exception as e:
            self.get_logger().error(f"Failed to load LSTM model: {e}")
            raise

    def _load_rl_agent(self):
        try:
            model = SAC.load(self.rl_model_path, device=self.device)
            self.get_logger().info("Successfully loaded SAC RL agent")
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

        # SYNC FIX: Subscribe to joint states from simulation (like real robot script)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states',  # MuJoCo publishes here
            self.joint_state_callback, 10
        )
        
        # Publisher: torque commands with correct QoS
        qos_profile = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        self.torque_cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_tau/torques_desired', qos_profile
        )
        
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, 'remote_robot/end_effector_pose', 10
        )
        
        # Control loop timer
        self.control_timer = self.create_timer(1.0/self.control_freq, self.control_loop)
        
        # Publisher timer
        self.state_publish_timer = self.create_timer(1.0/self.publish_freq, self.publish_end_effector_state)
    
    def _init_delay_simulation(self):
        """Initialize delay simulation."""
        self.experiment_config = 3  # Match your config
                
        self.delay_simulator = DelaySimulator(
            control_freq=self.control_freq,
            experiment_config=self.experiment_config
        )
        
        # Position history buffer for observation delay
        max_delay_steps = 1000
        self.position_history = deque(maxlen=max_delay_steps)

        self.get_logger().info(f"DelaySimulator initialized with config {self.experiment_config}")

    def _get_current_ee_pose_from_mujoco(self):
        """SYNC FIX: Use current joint positions to compute EE pose (like real robot script)."""
        if not np.all(np.isfinite(self.real_joint_positions)):
            self.get_logger().warn("Invalid joint positions for FK computation")
            return None, None
        
        # SYNC FIX: Update MuJoCo model with current joint positions
        self.kinematics_data.qpos[:self.num_joints] = self.real_joint_positions
        mujoco.mj_forward(self.kinematics_model, self.kinematics_data)
        
        # Get end-effector pose
        ee_body_name = 'panda_hand'
        ee_id = self.kinematics_model.body(ee_body_name).id
        position = self.kinematics_data.xpos[ee_id].copy()
        orientation_matrix = self.kinematics_data.xmat[ee_id].copy().reshape(3, 3)
        
        # Apply TCP offset
        tcp_offset_rotated = orientation_matrix @ self.ee_offset
        final_position = position + tcp_offset_rotated
        
        return final_position, orientation_matrix
        
    def cartesian_pose_callback(self, msg):
        """Store incoming leader positions in history buffer."""
        target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.position_history.append(target_position)

    def joint_state_callback(self, msg):
        """SYNC FIX: Update robot state from simulation and maintain sync (like real robot script)."""
        try:
            name_to_idx_map = {name: i for i, name in enumerate(msg.name)}
            
            for i, name in enumerate(self.joint_names):
                if name in name_to_idx_map:
                    idx = name_to_idx_map[name]
                    self.real_joint_positions[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self.real_joint_velocities[i] = msg.velocity[idx]
            
            # SYNC FIX: Update history buffers like real robot script
            self.joint_pos_history.append(self.real_joint_positions.copy())
            self.joint_vel_history.append(self.real_joint_velocities.copy())
                                
            if not self.robot_connected:
                self.current_q_target = self.real_joint_positions.copy()
                self.robot_connected = True
                self.get_logger().info("First joint state received. Robot connected and controller active!")
                
        except Exception as e:
            self.get_logger().error(f"Error processing joint states: {e}")

    def _get_delayed_position(self):
        """Get delayed position from history buffer."""
        if not self.position_history:
            return None
        
        delay_steps = self.delay_simulator.get_observation_delay_steps(len(self.position_history))
        
        if delay_steps >= len(self.position_history):
            return self.position_history[0]
            
        delay_index = len(self.position_history) - 1 - delay_steps
        return self.position_history[delay_index]

    def _get_rl_observation(self) -> np.ndarray:
        """Construct observation vector for RL agent."""
        joint_pos_hist_flat = np.array(list(self.joint_pos_history)).flatten()
        joint_vel_hist_flat = np.array(list(self.joint_vel_history)).flatten()
        action_history_flat = np.array(list(self.action_history)).flatten()

        obs_delay_steps = self.delay_simulator.get_observation_delay_steps(len(self.position_history))
        newest_available_idx = len(self.position_history) - 1 - obs_delay_steps
        newest_available_idx = max(0, newest_available_idx)
        
        lstm_input_sequence = []
        for i in range(self.lstm_sequence_length):
            idx = newest_available_idx - i
            pad_value = self.position_history[0] if self.position_history else np.zeros(3)
            lstm_input_sequence.insert(0, self.position_history[idx] if idx >= 0 else pad_value)
        
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(np.array(lstm_input_sequence)).unsqueeze(0).to(self.device)
            predicted_leader_pos_tensor = self.state_predictor(sequence_tensor)
            predicted_leader_pos = predicted_leader_pos_tensor.squeeze().cpu().numpy()

        num_raw_points = self.past_leader_points_in_obs
        raw_leader_pos_hist = np.array(lstm_input_sequence[-num_raw_points:]).flatten()

        return np.concatenate([
            joint_pos_hist_flat, joint_vel_hist_flat,
            predicted_leader_pos, raw_leader_pos_hist,
            action_history_flat
        ]).astype(np.float32)
    
    def control_loop(self):
        """SYNC FIX: Main control loop with IK-visualizer synchronization (like real robot script)."""
        
        if not self.robot_connected or not self.position_history:
            if self.robot_connected:
                self.current_q_target = self.real_joint_positions.copy()
            return

        self.debug_counter += 1

        # Get RL action for compensation
        observation = self._get_rl_observation()
        rl_action, _ = self.rl_model.predict(observation, deterministic=True)
        self.action_history.append(rl_action.copy())
        compensation_torque = rl_action * self.characteristic_torque
        
        # Get delayed Cartesian position
        delayed_position = self._get_delayed_position()
        if delayed_position is None:
            return

        # SYNC FIX: IK solver uses current joint positions as seed (like real robot script)
        q_goal, ik_error = self.ik_solver.solver(
            delayed_position,
            self.real_joint_positions,  # Use current state as seed for consistency
            ee_body_name='panda_hand'
        )
        if q_goal is None:
            if self.debug_counter % self.debug_freq == 0:
                self.get_logger().warn(f"IK failed. Holding target. Error: {ik_error:.4f}")
            q_goal = self.current_q_target.copy()

        # SYNC FIX: Smooth trajectory via joint space interpolation (like real robot script)
        q_error = q_goal - self.current_q_target
        error_norm = np.linalg.norm(q_error)
        
        if error_norm > self.max_q_step:
            step = q_error / error_norm * self.max_q_step
            self.current_q_target += step
        elif error_norm > 1e-6:
            self.current_q_target = q_goal.copy()
        
        q_target = self.current_q_target

        # SYNC FIX: Velocity filtering (like real robot script)
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

        # SYNC FIX: Action delay compensation (like real robot script)
        action_delay_steps = self.delay_simulator.get_action_delay_steps(len(self.position_history))
        action_delay_seconds = action_delay_steps / self.control_freq
        
        # Predict future joint state to compensate for action delay
        predicted_joint_positions = self.real_joint_positions + self.real_joint_velocities * action_delay_seconds
        predicted_joint_velocities = self.real_joint_velocities

        # PD controller
        pd_torques = self.pd_controller.compute_desired_acceleration(
            target_positions=q_target,
            target_velocities=self.smoothed_q_dot_target,
            current_positions=predicted_joint_positions,
            current_velocities=predicted_joint_velocities
        )
        
        # Combine PD torques with RL compensation
        final_torques = pd_torques
        torques_clipped = np.clip(final_torques, -self.torque_limits, self.torque_limits)
        
        # Publish torque commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = torques_clipped.tolist()
        self.torque_cmd_pub.publish(cmd_msg)

        # ADDED: Print final torques every control cycle for debugging
        print(f"Final Torques: [{torques_clipped[0]:+6.2f}, {torques_clipped[1]:+6.2f}, {torques_clipped[2]:+6.2f}, {torques_clipped[3]:+6.2f}, {torques_clipped[4]:+6.2f}, {torques_clipped[5]:+6.2f}, {torques_clipped[6]:+6.2f}] Nm")

        # Debug logging (like real robot script)
        if self.debug_counter % self.debug_freq == 0:
            obs_delay_val = self.delay_simulator.get_observation_delay_steps(len(self.position_history))
            self.get_logger().info(f"Delay Steps -> Obs: {obs_delay_val}, Action: {action_delay_steps}")
            self.get_logger().info(f"Target Vel Norm: {np.linalg.norm(self.smoothed_q_dot_target):.3f}")
            self.get_logger().info(f"Torques Norm: {np.linalg.norm(torques_clipped):.2f}")
            # ADDED: Also log individual torque values in ROS log
            self.get_logger().info(f"Individual Torques: {[f'{t:+6.2f}' for t in torques_clipped]}")

    def publish_end_effector_state(self):
        """SYNC FIX: Publish EE state using synchronized MuJoCo model (like real robot script)."""
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
        node = RemoteRobotController()
        node.get_logger().info("Remote robot controller with IK-Visualizer sync started")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down remote robot controller...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()