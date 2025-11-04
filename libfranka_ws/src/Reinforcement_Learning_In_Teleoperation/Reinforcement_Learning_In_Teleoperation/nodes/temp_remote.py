"""
Remote robot node using ROS2 (Follower).

The node implements:
- Subscribes to real robot's joint state
- Subscribes to desired robot's joint state
- execute Inverse dynamics + PD control law
- implement RL tau compensation
- Publishes the command to the real robot
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
import torch
from stable_baselines3 import SAC
from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_MUJOCO_MODEL_PATH, 
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    TORQUE_LIMITS,
    DEFAULT_KD_REMOTE,
    DEFAULT_KP_REMOTE,
)

class RemoteRobotNode(Node):
    def __init__(self):
        super().__init__('remote_robot_node')
        
        # Initialize parameters
        self.n_joints_ = N_JOINTS
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.kp_ = DEFAULT_KP_REMOTE
        self.kd_ = DEFAULT_KD_REMOTE
        self.torque_limits_ = TORQUE_LIMITS
        self.joint_names_ = [f'panda_joint{i+1}' for i in range(self.n_joints_)]
        self.initial_joint_config_ = INITIAL_JOINT_CONFIG
        self.current_q_ = self.initial_joint_config_.copy()
        self.current_dq_ = np.zeros(self.n_joints_, dtype=np.float32)
        
        # Initialize Mujoco model and data
        model_path = DEFAULT_MUJOCO_MODEL_PATH
        self.mj_model_ = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data_ = mujoco.MjData(self.mj_model_)
        
        # Latest command from AgentNode
        self.target_q_ = INITIAL_JOINT_CONFIG.copy()
        self.target_qd_ = np.zeros(self.n_joints_)
        self.tau_rl_ = np.zeros(self.n_joints_)
        
        # Readiness flag
        self.robot_state_ready_ = False
        self.agent_command_ready_ = False
        
        # ROS2 Interfaces
        # Subscribe to agent command
        self.agent_command_sub_ = self.create_subscription(
            JointState, 'agent/command', self.agent_command_callback, 10)
        
        # Subscribe to real robot joint states
        self.robot_state_sub_ = self.create_subscription(
            JointState, 'remote_robot/joint_states', self.robot_state_callback, 10)
        
        # Publisher to the real robot's torque driver
        self.torque_command_pub_ = self.create_publisher(
            Float64MultiArray, '/joint_tau/torques_desired', 10)
        
        # Main control loop timer
        self.control_timer_ = self.create_timer(
            self.dt_, self.control_loop_callback)
        
        self.get_logger().info("Remote Robot Node (Follower) initialized.")
        
    def agent_command_callback(self, msg: JointState) -> None:
        """
        Receives the command from the AgentNode.
        - position = predicted target q
        - velocity = predicted target qd
        - effort = RL torque compensation tau_rl
        """
        try:
            # Re-order logic from AgentNode
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            self.target_q_ = np.array([msg.position[name_to_index_map[name]] for name in self.joint_names_])
            self.target_qd_ = np.array([msg.velocity[name_to_index_map[name]] for name in self.joint_names_])
            self.tau_rl_ = np.array([msg.effort[name_to_index_map[name]] for name in self.joint_names_])

            if not self.agent_command_ready_:
                self.agent_command_ready_ = True
                self.get_logger().info("First command from AgentNode received.")
                
        except (KeyError, IndexError) as e:
            self.get_logger().warn(f"Error processing agent command: {e}")

    def robot_state_callback(self, msg: JointState) -> None:
        """Receives the real-time state from the robot hardware."""
        try:
            # Assuming msg.name is in the correct order or matches joint_names_
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            self.current_q_ = np.array([msg.position[name_to_index_map[name]] for name in self.joint_names_])
            self.current_qd_ = np.array([msg.velocity[name_to_index_map[name]] for name in self.joint_names_])
            
            if not self.robot_state_ready_:
                self.robot_state_ready_ = True
                self.get_logger().info("First hardware state from Remote Robot received.")
                
        except (KeyError, IndexError) as e:
            self.get_logger().warn(f"Error processing robot state: {e}")

    def _compute_inverse_dynamics_torque(
        self,
        q: NDArray[np.float64],
        qd: NDArray[np.float64],
        qdd: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Inverse dynamics using MuJoCo's built-in function."""
        
        # Save current state
        qpos_save = self.mj_data_.qpos.copy()
        qvel_save = self.mj_data_.qvel.copy()
        qacc_save = self.mj_data_.qacc.copy()

        # Set desired state
        self.mj_data_.qpos[:self.n_joints_] = q
        self.mj_data_.qvel[:self.n_joints_] = qd
        self.mj_data_.qacc[:self.n_joints_] = qdd

        # Compute inverse dynamics
        mujoco.mj_inverse(self.mj_model_, self.mj_data_)
        tau = self.mj_data_.qfrc_inverse[:self.n_joints_].copy()

        # Restore original state
        self.mj_data_.qpos[:] = qpos_save
        self.mj_data_.qvel[:] = qvel_save
        self.mj_data_.qacc[:] = qacc_save
        
        return tau

    def control_loop_callback(self) -> None:
        """Main control loop for the remote robot."""
        
        # Wait until both agent and robot are ready
        if not self.agent_command_ready_ or not self.robot_state_ready_:
            self.get_logger().warn(
                "Waiting for agent command and robot state...",
                throttle_duration_sec=5.0
            )
            return

        try:
            # Get current and target states
            q_current = self.current_q_
            qd_current = self.current_qd_
            
            q_target = self.target_q_     # From agent
            qd_target = self.target_qd_   # From agent
            tau_rl = self.tau_rl_       # From agent
            
            # self.get_logger().info(f"Current State: {q_current}, {qd_current}")
            # self.get_logger().info(f"Target State: {q_target}, {qd_target}, Tau_RL: {tau_rl}")
            # self.get_logger().info(f"tau_rl: {tau_rl}")
           
            # Apply PD Controller
            q_error = q_target - q_current
            qd_error = qd_target - qd_current
            qdd_desired = self.kp_ * q_error + self.kd_ * qd_error # Desired acceleration
            
            # Inverse Dynamics (Baseline Torque)
            # Computes M(q) * qdd_desired + C(q, qd)
            tau_baseline = self._compute_inverse_dynamics_torque(
                q=q_current,
                qd=qd_current,
                qdd=qdd_desired,
            )
            
            # Total Command = Baseline (ID+PD) + RL Compensation
            tau_command = tau_baseline + tau_rl

            # Torque Clipping
            tau_clipped = np.clip( tau_command, -self.torque_limits_, self.torque_limits_)

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
    except Exception as e:
        print(f"Node failed to initialize or run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if remote_robot_node:
            remote_robot_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()