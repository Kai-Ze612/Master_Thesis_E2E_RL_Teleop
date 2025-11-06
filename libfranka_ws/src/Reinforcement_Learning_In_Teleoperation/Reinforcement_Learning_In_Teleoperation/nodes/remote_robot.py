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
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
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
        self.kp_ = DEFAULT_KP_REMOTE / 50
        self.kd_ = DEFAULT_KD_REMOTE / 50
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
        self.current_tau_rl_ = np.zeros(self.n_joints_)

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
            seed=50 # Fixed seed for reproducibility
        )
        
        # Pre-fill action buffer to avoid initial empty buffer issues
        self.torque_command_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self._prefill_action_buffer()
        
        # State flags
        self.robot_state_ready_ = False
        self.target_command_ready_ = False
        self.tau_rl_ready_ = False
        self.local_state_ready_ = False
        
        # ROS2 Interfaces
        
        # Subscriber to target joint commands from agent
        self.target_command_sub_ = self.create_subscription(
            JointState, 'agent/predict_target', self.target_command_callback, 10
        )
        
        # Subscription to the AGENT's compensation torque
        self.tau_rl_sub_ = self.create_subscription(
            Float64MultiArray, 'agent/tau_rl', self.tau_rl_callback, 10
        )
        
        #  Subscription to the REAL ROBOT's state (real time)
        self.robot_state_sub_ = self.create_subscription(
            JointState,  'remote_robot/joint_states', self.robot_state_callback, 10
        )
        
        # Subscribe to local robot state in real time for debugging
        self.local_robot_state_sub_ = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 10
        )
        
        # Publisher to the ROBOT CONTROLLER
        controller_command_topic = 'joint_tau/torques_desired' # Will be remapped by launch file
        self.get_logger().info(f"Publishing torque commands to RELATIVE topic: {controller_command_topic}")
        
        self.torque_command_pub_ = self.create_publisher(
            Float64MultiArray,  controller_command_topic, 10)
        
        self.control_timer_ = self.create_timer(
            self.dt_, self.control_loop_callback)
        
        self.get_logger().info("Remote Robot Node initialized.")
    
    def _prefill_action_buffer(self) -> None:
        """Prefill the action history buffer with zeros."""
        num_prefill_steps = int(WARM_UP_DURATION * self.control_freq_)
        zeros_action = np.zeros(self.n_joints_)
        for _ in range(num_prefill_steps):
            self.torque_command_history_.append(zeros_action)
        self.get_logger().info(f"Pre-filled action buffer with {num_prefill_steps} zero actions.")

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
        
        # Wait for all required data
        if not self.robot_state_ready_:
             self.get_logger().warn(
                "Waiting for Robot State...",
                throttle_duration_sec=5.0
            )
             return

        # Wait for agent commands
        if not self.target_command_ready_ or not self.tau_rl_ready_:
            self.get_logger().warn(
                "Waiting for Agent Commands (predict_target, tau_rl)... Publishing G-Comp + PD to hold.",
                throttle_duration_sec=5.0
            )
            # Use default hold targets if agent isn't ready
            q_target = self.target_q_ 
            qd_target = self.target_qd_
            tau_rl = self.current_tau_rl_ # This will be zeros until first msg
        else:
            # Agent is ready, use its commands
            q_target = self.target_q_ 
            qd_target = self.target_qd_
            tau_rl = self.current_tau_rl_
        
        # Start the control loop
        try:
            q_current = self.current_q_
            qd_current = self.current_qd_
            
            # PD Calculation
            tau_gravity = self._compute_gravity_compensation(q_current)
            q_error = self._normalize_angle(q_target - q_current)
            qd_error = qd_target - qd_current
            tau_pd = self.kp_ * q_error + self.kd_ * qd_error
            
            # Final Torque Command
            tau_command = tau_gravity + tau_pd + tau_rl
            tau_clipped = np.clip(tau_command, -self.torque_limits_, self.torque_limits_)

            # ACTION DELAY
            # Store the calculated (non-delayed) command
            self.torque_command_history_.append(tau_clipped)

            # Get the delay steps from the simulator
            history_len = len(self.torque_command_history_)
            action_delay_steps = self.action_delay_simulator_.get_action_delay_steps()

            # Get the delayed command from history
            delayed_action_idx = -(action_delay_steps + 1)
            safe_idx = np.clip(delayed_action_idx, -history_len, -1)
            torque_to_publish = self.torque_command_history_[safe_idx]
            
            # Publish the DELAYED Command to Hardware
            msg = Float64MultiArray()
            msg.data = torque_to_publish.tolist()
            self.torque_command_pub_.publish(msg)

            # print out q actual and q predicted for monitoring
            local_q_for_logging = np.zeros(self.n_joints_)
            if self.local_state_ready_:
                local_q_for_logging = self.current_local_q_

            # Format vectors for printing
            q_pred_str = np.round(q_target, 2) # Agent's prediction
            q_local_str = np.round(local_q_for_logging, 2) # Local robot's actual state

            self.get_logger().info(
                f"Q_Pred (Agent): {q_pred_str}\n"
                f"Q_Actual (Local): {q_local_str}",
                throttle_duration_sec=2.0 # Throttled to 2 seconds for readability
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
        if remote_robot_node:
            remote_robot_node.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        if remote_robot_node:
            remote_robot_node.destroy_node()

if __name__ == '__main__':
    main()