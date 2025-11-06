"""
The script is the remote robot, using ROS2 node

The remote robot subscribes to agent published predicted trajectory and compensation tau

before scribe to agent command
"""

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Mujoco imports
import mujoco

# Python imports
import numpy as np
from numpy.typing import NDArray

# Custom imports
from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_MUJOCO_MODEL_PATH, 
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    TORQUE_LIMITS,
    DEFAULT_KD_REMOTE,
    DEFAULT_KP_REMOTE,
    TCP_OFFSET,
    EE_BODY_NAME,
)

class RemoteRobotNode(Node):
    def __init__(self):
        super().__init__('remote_robot_node')
        
        # Initialize parameters from config.py
        self.n_joints_ = N_JOINTS
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.tcp_offset_ = TCP_OFFSET
        self.kp_ = DEFAULT_KP_REMOTE
        self.kd_ = DEFAULT_KD_REMOTE
        self.torque_limits_ = TORQUE_LIMITS
        self.joint_names_ = [f'panda_joint{i+1}' for i in range(self.n_joints_)]
        self.initial_joint_config_ = INITIAL_JOINT_CONFIG
        self.ee_body_name_ = EE_BODY_NAME
        
        # Initialize remote robot current joint states and velocities
        self.current_q_ = self.initial_joint_config_.copy()
        self.current_qd_ = np.zeros(self.n_joints_, dtype=np.float32)

        # Initialize Mujoco model and data
        model_path = DEFAULT_MUJOCO_MODEL_PATH
        self.mj_model_ = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data_ = mujoco.MjData(self.mj_model_)
        self.ee_body_id_ = self.mj_model_.body(name=self.ee_body_name_).id
        
        # Initialize target joint states and velocities
        self.target_q_ = INITIAL_JOINT_CONFIG.copy()
        self.target_qd_ = np.zeros(self.n_joints_)

        # State flags
        self.robot_state_ready_ = False
        self.target_command_ready_ = False 
        
        # ROS2 Interfaces
        self.target_command_sub_ = self.create_subscription(
            JointState, 
            'local_robot/joint_states', 
            self.target_command_callback, 
            10
        )
        
        self.robot_state_sub_ = self.create_subscription(
            JointState, '/franka/joint_states', self.robot_state_callback, 10)
        
        controller_command_topic = '/joint_tau/torques_desired'
        self.get_logger().info(f"Publishing torque commands to ABSOLUTE topic: {controller_command_topic}")
        
        self.torque_command_pub_ = self.create_publisher(
            Float64MultiArray, 
            controller_command_topic,
            10
        )
        
        self.control_timer_ = self.create_timer(
            self.dt_, self.control_loop_callback)
        
        self.get_logger().info("Remote Robot Node (Follower) [PD+Gravity+EE_POS] initialized.")
        self.get_logger().info(f"Kp = {self.kp_}, Kd = {self.kd_}")

    def target_command_callback(self, msg: JointState) -> None:
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            self.target_q_ = np.array([msg.position[name_to_index_map[name]] for name in self.joint_names_])
            self.target_qd_ = np.array([msg.velocity[name_to_index_map[name]] for name in self.joint_names_])

            if not self.target_command_ready_:
                self.target_command_ready_ = True
                self.get_logger().info("First command from Local Robot received.")

            self.get_logger().info(
                f"Target q: [{self.target_q_[0]:.3f}, {self.target_q_[1]:.3f}, {self.target_q_[2]:.3f}, "
                f"{self.target_q_[3]:.3f}, {self.target_q_[4]:.3f}, {self.target_q_[5]:.3f}, {self.target_q_[6]:.3f}]",
                throttle_duration_sec=2.0
            )

        except (KeyError, IndexError) as e:
            self.get_logger().warn(f"Error processing target command: {e}")

    def robot_state_callback(self, msg: JointState) -> None:
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            self.current_q_ = np.array([msg.position[name_to_index_map[name]] for name in self.joint_names_])
            self.current_qd_ = np.array([msg.velocity[name_to_index_map[name]] for name in self.joint_names_])
            
            if not self.robot_state_ready_:
                self.robot_state_ready_ = True
                self.get_logger().info("First hardware state from Remote Robot received.")
                self.get_logger().info(
                    f"Initial q: [{self.current_q_[0]:.3f}, {self.current_q_[1]:.3f}, {self.current_q_[2]:.3f}, "
                    f"{self.current_q_[3]:.3f}, {self.current_q_[4]:.3f}, {self.current_q_[5]:.3f}, {self.current_q_[6]:.3f}]"
                )
        except (KeyError, IndexError) as e:
            self.get_logger().warn(f"Error processing robot state: {e}")

    def _normalize_angle(self, angle: np.ndarray) -> np.ndarray:
        """Normalize an angle or array of angles to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
        
    def _compute_gravity_compensation(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes the torque needed to counteract gravity."""
        
        qpos_save = self.mj_data_.qpos.copy()
        qvel_save = self.mj_data_.qvel.copy()
        qacc_save = self.mj_data_.qacc.copy()

        self.mj_data_.qpos[:self.n_joints_] = q
        self.mj_data_.qvel[:self.n_joints_] = 0.0
        self.mj_data_.qacc[:self.n_joints_] = 0.0

        mujoco.mj_inverse(self.mj_model_, self.mj_data_)
        tau_gravity = self.mj_data_.qfrc_inverse[:self.n_joints_].copy()

        self.mj_data_.qpos[:] = qpos_save
        self.mj_data_.qvel[:] = qvel_save
        self.mj_data_.qacc[:] = qacc_save
        
        return tau_gravity

    def _compute_tcp_position(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute TCP (Tool Center Point) position using forward kinematics."""
        
        # Save current state
        qpos_save = self.mj_data_.qpos.copy()
        
        # Set joint positions
        self.mj_data_.qpos[:self.n_joints_] = q
        
        # Compute forward kinematics
        mujoco.mj_forward(self.mj_model_, self.mj_data_)
        
        # Get flange position and orientation matrix
        flange_pos = self.mj_data_.xpos[self.ee_body_id_].copy()
        flange_mat = self.mj_data_.xmat[self.ee_body_id_].copy().reshape(3, 3)
        
        # Transform TCP offset from flange frame to world frame
        tcp_offset_world = flange_mat @ self.tcp_offset_  # ✓ Applies TCP offset
        
        # TCP position in world frame
        tcp_pos = flange_pos + tcp_offset_world  # ✓ Final TCP position
        
        # Restore state
        self.mj_data_.qpos[:] = qpos_save
        
        return tcp_pos  # ✓ Returns TCP center, matching IK target
    
    def control_loop_callback(self) -> None:
        
        if not self.robot_state_ready_:
            self.get_logger().warn(
                "Waiting for robot state... Cannot compute G-Comp.",
                throttle_duration_sec=5.0
            )
            return
        
        if not self.target_command_ready_:
            self.get_logger().warn(
                "Target command not yet received. Publishing G-Comp + PD to hold initial position.",
                throttle_duration_sec=5.0
            )
        
        try:
            q_current = self.current_q_
            qd_current = self.current_qd_
            q_target = self.target_q_ 
            
            qd_target_for_damping = np.zeros(self.n_joints_) 
            tau_rl = np.zeros(self.n_joints_) 
            
            # Compute control components
            tau_gravity = self._compute_gravity_compensation(q_current)
            q_error_unnorm = q_target - q_current
            q_error = self._normalize_angle(q_error_unnorm)
            qd_error = qd_target_for_damping - qd_current
            tau_pd = self.kp_ * q_error + self.kd_ * qd_error
            
            tau_command = tau_gravity + tau_pd + tau_rl
            tau_clipped = np.clip(tau_command, -self.torque_limits_, self.torque_limits_)

            # Compute end-effector position
            ee_pos = self._compute_tcp_position(q_current)
            
            # Check for NaN
            if np.isnan(tau_clipped).any():
                self.get_logger().error("NaN detected in torque command! Not publishing.")
                return 
            
            # Display end-effector position (always, no throttling)
            self.get_logger().info(
                f"EE Position: X={ee_pos[0]:.4f}, Y={ee_pos[1]:.4f}, Z={ee_pos[2]:.4f}",
                throttle_duration_sec=0.5
            )
            
            # Detailed logging (throttled)
            self.get_logger().info(
                f"Current q: [{q_current[0]:.3f}, {q_current[1]:.3f}, {q_current[2]:.3f}, "
                f"{q_current[3]:.3f}, {q_current[4]:.3f}, {q_current[5]:.3f}, {q_current[6]:.3f}]",
                throttle_duration_sec=2.0
            )
            
            self.get_logger().info(
                f"q_error: [{q_error[0]:.3f}, {q_error[1]:.3f}, {q_error[2]:.3f}, "
                f"{q_error[3]:.3f}, {q_error[4]:.3f}, {q_error[5]:.3f}, {q_error[6]:.3f}]",
                throttle_duration_sec=2.0
            )
            
            self.get_logger().info(
                f"Tau_gravity: [{tau_gravity[0]:.2f}, {tau_gravity[1]:.2f}, {tau_gravity[2]:.2f}, "
                f"{tau_gravity[3]:.2f}, {tau_gravity[4]:.2f}, {tau_gravity[5]:.2f}, {tau_gravity[6]:.2f}]",
                throttle_duration_sec=2.0
            )
            
            self.get_logger().info(
                f"Tau_PD: [{tau_pd[0]:.2f}, {tau_pd[1]:.2f}, {tau_pd[2]:.2f}, "
                f"{tau_pd[3]:.2f}, {tau_pd[4]:.2f}, {tau_pd[5]:.2f}, {tau_pd[6]:.2f}]",
                throttle_duration_sec=2.0
            )
            
            self.get_logger().info(
                f"Tau_total: [{tau_clipped[0]:.2f}, {tau_clipped[1]:.2f}, {tau_clipped[2]:.2f}, "
                f"{tau_clipped[3]:.2f}, {tau_clipped[4]:.2f}, {tau_clipped[5]:.2f}, {tau_clipped[6]:.2f}]",
                throttle_duration_sec=1.0
            )

            # Publish Command to Hardware
            msg = Float64MultiArray()
            msg.data = tau_clipped.tolist()
            self.torque_command_pub_.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error in remote control loop: {e}")
            import traceback
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    remote_robot_node = None
    try:
        remote_robot_node = RemoteRobotNode()
        rclpy.spin(remote_robot_node)
    except KeyboardInterrupt:
        if remote_robot_node:
            remote_robot_node.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        if remote_robot_node:
            remote_robot_node.destroy_node()
        # rclpy.shutdown() # Removed to prevent crash on exit

if __name__ == '__main__':
    main()