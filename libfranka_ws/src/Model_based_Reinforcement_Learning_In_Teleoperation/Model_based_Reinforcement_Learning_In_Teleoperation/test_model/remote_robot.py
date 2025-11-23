"""
The script is the remote robot, using ROS2 node

Pipelines:
1. subscribe to 'agent/predicted_target' (JointState) for desired joint positions and velocities
2. subscribe to 'agent/compensation_tau' (Float64MultiArray) for torque compensation
3. PD control loop with gravity compensation to determine torque baseline
4. adding torque baseline with compensation tau from agent
5. adding action delay
6. publish to '/joint_tau/torques_desired' (Float64MultiArray) for robot control
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
from collections import deque

# Custom imports
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_MUJOCO_MODEL_PATH, 
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    TORQUE_LIMITS,
    DEFAULT_KD_REMOTE,
    DEFAULT_KP_REMOTE,
    TCP_OFFSET,
    EE_BODY_NAME,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
    WARM_UP_DURATION,
)

class RemoteRobotNode(Node):
    def __init__(self):
        super().__init__('remote_robot_node')
        
        # Initialize parameters from config.py
        self.n_joints_ = N_JOINTS
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.tcp_offset_ = TCP_OFFSET
        self.kp_ = DEFAULT_KP_REMOTE / 2
        self.kd_ = DEFAULT_KD_REMOTE / 2
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
        self.target_qd_ = np.zeros(self.n_joints_)  # Store target velocity
        self.current_tau_rl_ = np.zeros(self.n_joints_)

        # Vairables for storing true local state (for debugging)
        self.current_local_q_ = INITIAL_JOINT_CONFIG.copy()
        self.current_local_qd_ = np.zeros(self.n_joints_)
        
        # Action delay
        self.default_experiment_config_ = ExperimentConfig.HIGH_DELAY.value
        self.declare_parameter('experiment_config', self.default_experiment_config_)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        try:
            self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        except ValueError:
            self.get_logger().fatal(f"Invalid 'experiment_config' int: {self.experiment_config_int_}")
            raise

        self.action_delay_simulator_ = DelaySimulator(
            control_freq=self.control_freq_,
            config=self.delay_config_,
            seed=50 
        )
        
        self.torque_command_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        
        # State flags
        self.robot_state_ready_ = False
        self.target_command_ready_ = False
        self.tau_rl_ready_ = False
        self.local_state_ready_ = False
        
        # ROS2 Interfaces
        self.target_command_sub_ = self.create_subscription(
            JointState, 'agent/predict_target', self.target_command_callback, 10
        )
        
        self.tau_rl_sub_ = self.create_subscription(
            Float64MultiArray, 'agent/tau_rl', self.tau_rl_callback, 10
        )
        
        self.robot_state_sub_ = self.create_subscription(
            JointState,  'remote_robot/joint_states', self.robot_state_callback, 10
        )
        
        self.local_robot_state_sub_ = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 10
        )
        
        controller_command_topic = 'joint_tau/torques_desired' 
        self.get_logger().info(f"Publishing torque commands to RELATIVE topic: {controller_command_topic}")
        
        self.torque_command_pub_ = self.create_publisher(
            Float64MultiArray,  controller_command_topic, 10)
        
        self.control_timer_ = self.create_timer(
            self.dt_, self.control_loop_callback)
        
        self.get_logger().info("Remote Robot Node initialized.")

    def target_command_callback(self, msg: JointState) -> None:
        """Callback for the AGENT's predicted target state."""
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            self.target_q_ = np.array([msg.position[name_to_index_map[name]] for name in self.joint_names_])
            self.target_qd_ = np.array([msg.velocity[name_to_index_map[name]] for name in self.joint_names_])
            
            if not self.target_command_ready_:
                self.target_command_ready_ = True
                self.get_logger().info("First command from Agent (predict_target) received.")

        except (KeyError, IndexError) as e:
            self.get_logger().warn(f"Error processing target command: {e}")

    def tau_rl_callback(self, msg: Float64MultiArray) -> None:
        """Callback for the AGENT's compensation torque."""
        self.current_tau_rl_ = np.array(msg.data, dtype=np.float32)
        if not self.tau_rl_ready_:
            self.tau_rl_ready_ = True
            self.get_logger().info("First command from Agent (tau_rl) received.")

    def robot_state_callback(self, msg: JointState) -> None:
        """Callback for the REAL ROBOT's current state."""
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            self.current_q_ = np.array([msg.position[name_to_index_map[name]] for name in self.joint_names_])
            self.current_qd_ = np.array([msg.velocity[name_to_index_map[name]] for name in self.joint_names_])
            
            if not self.robot_state_ready_:
                self.robot_state_ready_ = True
                self.get_logger().info("First hardware state from Remote Robot received.")
        except (KeyError, IndexError) as e:
            self.get_logger().warn(f"Error processing robot state: {e}")
            
    def local_robot_state_callback(self, msg: JointState) -> None:
        """Callback for the LOCAL ROBOT's current state."""
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            self.current_local_q_ = np.array([msg.position[name_to_index_map[name]] for name in self.joint_names_])
            self.current_local_qd_ = np.array([msg.velocity[name_to_index_map[name]] for name in self.joint_names_])
            
            if not self.local_state_ready_:
                self.local_state_ready_ = True
                self.get_logger().info("First state from Local Robot received.")
        except (KeyError, IndexError) as e:
            self.get_logger().warn(f"Error processing LOCAL robot state: {e}")
        
    def _normalize_angle(self, angle: np.ndarray) -> np.ndarray:
        """Normalize an angle or array of angles to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _get_inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a_desired: np.ndarray) -> np.ndarray:
        """
        Compute Torque using MuJoCo Inverse Dynamics.
        Equation: tau = M(q)*a_desired + C(q,v)*v + G(q)
        """
        # 1. Update MuJoCo with the CURRENT Robot State
        # We must use the current state so M(q) and G(q) match reality
        self.mj_data_.qpos[:self.n_joints_] = q
        self.mj_data_.qvel[:self.n_joints_] = v
        
        # 2. Set the DESIRED Acceleration (from PD)
        self.mj_data_.qacc[:self.n_joints_] = a_desired

        # 3. Compute Inverse Dynamics
        # MuJoCo solves: tau = M*qacc + C*qvel + G
        mujoco.mj_inverse(self.mj_model_, self.mj_data_)

        return self.mj_data_.qfrc_inverse[:self.n_joints_].copy()

    def control_loop_callback(self) -> None:
        
        # Wait for all required data
        if not self.robot_state_ready_ or not self.target_command_ready_ or not self.tau_rl_ready_ or not self.local_state_ready_:
            self.get_logger().warn(f"Waiting for data...", throttle_duration_sec=2.0)
            return

        q_target = self.target_q_ 
        qd_target = self.target_qd_ 
        
        try:
            q_current = self.current_q_
            qd_current = self.current_qd_
            
            # PD Control now outputs Acceleration ---
            q_error = self._normalize_angle(q_target - q_current)
            qd_error = qd_target - qd_current
            
            # acc_desired is in rad/s^2
            # stiffness in Acceleration space, not Torque space.
            acc_desired = self.kp_ * q_error + self.kd_ * qd_error
            
            # Use Computed Torque (Inverse Dynamics)
            # This calculates: tau = M(q)*acc_desired + C(q,qd)*qd + G(q)
            tau_id = self._get_inverse_dynamics(q_current, qd_current, acc_desired)
            
            # RL compensation
            tau_rl = self.current_tau_rl_
            
            # Safety: No torque on last joint
            tau_id[-1] = 0.0 
            
            # Final Command
            tau_command = tau_id + tau_rl * 0
            tau_clipped = np.clip(tau_command, -self.torque_limits_, self.torque_limits_)

            # Apply action delay
            self.torque_command_history_.append(tau_clipped)
            
            action_delay_steps = self.action_delay_simulator_.get_action_delay_steps()
            
            if action_delay_steps >= len(self.torque_command_history_):
                torque_to_publish = np.zeros(self.n_joints_)
            else:
                torque_to_publish = self.torque_command_history_[-1 - action_delay_steps]
            
            # Publish
            msg = Float64MultiArray()
            msg.data = torque_to_publish.tolist()
            self.torque_command_pub_.publish(msg)

            self.get_logger().info(
                f"\n--- DEBUG (throttled 0.1s) ---\n"
                f"True q:       {np.round(self.current_local_q_, 3)}\n"
                f"Predicted q:  {np.round(q_target, 3)}\n"
                f"Remote q:     {np.round(q_current, 3)}\n"
                f"Tau ID:       {np.round(tau_id, 3)} (Includes Gravity)\n"
                f"Tau RL:       {np.round(tau_rl, 3)}\n",
                throttle_duration_sec=0.1
            )

        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            import traceback
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    remote_robot_node = None
    try:
        remote_robot_node = RemoteRobotNode()
        rclpy.spin(remote_robot_node)
    except KeyboardInterrupt:
        pass
    finally:
        if remote_robot_node:
            remote_robot_node.destroy_node()

if __name__ == '__main__':
    main()