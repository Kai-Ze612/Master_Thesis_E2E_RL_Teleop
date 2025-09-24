"""
Local Leader Robot ROS2 Node leader_mujoco_simulator
Robot model: Franka Panda

This node will receive positional command, use IK to compute joint positions and call PD controller to calculate torques.

Then the calculated torques will be published to Mujoco.
"""

# ROS2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
from ament_index_python.packages import get_package_share_directory
from multi_mode_control_msgs.msg import JointGoal
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# MuJoCo imports
import mujoco

# Python imports
import numpy as np
import os

# Custom IK imports
from Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays.controllers.inverse_kinematics import InverseKinematicsSolver

class LocalRobot(Node):
    def __init__(self):
        super().__init__('local_controller_node')
        self.get_logger().info("Starting initialization...")
    
        self._init_parameters()
        self._init_ik_solver()
        self._init_ros_interfaces()

        self.is_active = False
        self.controller_ready = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)  
        
        self.get_logger().info("Local Robot Node ready - waiting for activation")

    def _init_parameters(self):
        """Initialize robot parameters and configuration."""
        
        # Setting publish freq as ros2 parameter
        self.declare_parameter('publish_freq', 50)
        
        self.publish_freq = self.get_parameter('publish_freq').value
        
        self.num_joints = 7
        self.joint_names = [f'panda_joint{i+1}' for i in range(self.num_joints)]
        
        self.joint_limits_lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.joint_limits_upper = [2.8973, 1.7628, 2.8973, -0.1518, 2.8973, 3.7525, 2.8973]
        
        franka_description_path = get_package_share_directory('franka_description')
        self.model_path = os.path.join(franka_description_path, 'mujoco', 'franka', 'scene.xml')
        
        self.virtual_joint_positions = None
        self.virtual_joint_velocities = None
        self.last_q_target = None
        self.last_time = None

    def _init_ik_solver(self):
        "IK solver initialization"
        self.mujoco_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.ik_solver = InverseKinematicsSolver(self.mujoco_model, self.joint_limits_lower, self.joint_limits_upper)
        self.get_logger().info("Inverse Kinematics solver is ready.")

    def _init_ros_interfaces(self):
        """Initialize ROS 2 subscribers and publishers."""
        # Start Command
        self.start_sub = self.create_subscription(
            Bool, '/local_robot/start_command', self.start_callback, 100)

        # Publisher: local_robot_ee_pose
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/local_robot/leader_pose', 100)

        # Subscriber to trajectory commands
        self.cartesian_cmd_sub = self.create_subscription(
            Point, '/local_robot/cartesian_commands', self.cartesian_command_callback, 100)

        # Subscriber to joint states for IK solver with inital guess
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Publish the solution joint position
        self.joint_goal_pub = self.create_publisher(
            JointGoal, '/panda/panda_joint_impedance_controller/desired_pose', 100)
        
        self.state_timer = self.create_timer(1.0/self.publish_freq, self.publish_states)
        self.get_logger().info("ROS2 interfaces initialized")

    def cartesian_command_callback(self, msg):
        """ Subscribe to positional command and use IK to find target joint position"""
        if not self.is_active or not self.controller_ready:
            return

        target_position = np.array([msg.x, msg.y, msg.z]) # x, y coordinate have something wrong
        joint_solution, _ = self.ik_solver.solver(target_position, self.virtual_joint_positions, 'panda_hand')
       
        if joint_solution is not None:
            safety_margin = 0.02
            limit = self.joint_limits_upper[3]
            if joint_solution[3] > (limit - safety_margin):
                joint_solution[3] = limit - safety_margin

            current_time = self.get_clock().now().nanoseconds / 1e9
            q_dot_target = np.zeros(self.num_joints) # Default to zero velocity
            
            if self.last_q_target is not None and self.last_time is not None:
                dt = current_time - self.last_time
                if dt > 1e-6:
                    q_dot_target = (joint_solution - self.last_q_target) / dt
            
            self.last_q_target = joint_solution.copy()
            self.last_time = current_time
            
            goal_msg = JointGoal()
            goal_msg.q = joint_solution.tolist()
            goal_msg.qd = q_dot_target.tolist()
            self.joint_goal_pub.publish(goal_msg)
            self.get_logger().info(f"IK solution found and published.")
        else:
            self.get_logger().warn(f"IK failed for target: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]")

    def joint_state_callback(self, msg):
        """Update current joint state from MuJoCo simulation."""
        if self.controller_ready:
            name_to_idx = {name: i for i, name in enumerate(msg.name)}
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in name_to_idx:
                    idx = name_to_idx[joint_name]
                    self.virtual_joint_positions[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self.virtual_joint_velocities[i] = msg.velocity[idx]
            return

        # This section only runs for the very first message to get initial state
        temp_positions = np.zeros(self.num_joints)
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        if all(jn in name_to_idx for jn in self.joint_names):
            for i, jn in enumerate(self.joint_names):
                idx = name_to_idx[jn]
                temp_positions[i] = msg.position[idx]
            
            self.virtual_joint_positions = temp_positions
            self.virtual_joint_velocities = np.zeros(self.num_joints)
            self.controller_ready = True
            self.get_logger().info("Controller is now ready with initial state.")
    
    def start_callback(self, msg: Bool):
        """Handle start/stop commands."""
        if msg.data and not self.is_active:
            self.is_active = True
            self.get_logger().info("Controller ACTIVATED")
        elif not msg.data and self.is_active:
            self.is_active = False
            self.get_logger().info("Controller DEACTIVATED")
    
    def publish_states(self):
        """Publish robot states by looking up the transform from the TF2 buffer."""
        target_frame = 'panda_hand'      # What we want the pose of
        source_frame = 'panda_link0'     # Frame we want pose expressed in
        
        try:
            # Get pose of panda_hand in panda_link0 frame
            t = self.tf_buffer.lookup_transform(
                source_frame,    # to_frame (reference frame)
                target_frame,    # from_frame (target object)
                rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'Could not transform {target_frame} to {source_frame}: {e}')
            return
            
        # Create and publish the PoseStamped message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = source_frame  # Pose is expressed in this frame
        
        # Position
        msg.pose.position.x = t.transform.translation.x
        msg.pose.position.y = t.transform.translation.y
        msg.pose.position.z = t.transform.translation.z
        
        # Orientation (was missing)
        msg.pose.orientation.x = t.transform.rotation.x
        msg.pose.orientation.y = t.transform.rotation.y
        msg.pose.orientation.z = t.transform.rotation.z
        msg.pose.orientation.w = t.transform.rotation.w
        
        self.ee_pose_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = LocalRobot()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()

# if __name__ == '__main__':
#     main()