import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor  # Required for safety
from rclpy.callback_groups import ReentrantCallbackGroup # Required for safety
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped

import mujoco
import numpy as np
from collections import deque

from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_MUJOCO_MODEL_PATH,
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    TORQUE_LIMITS,
    MAX_TORQUE_COMPENSATION,
    DEFAULT_KD_REMOTE,
    DEFAULT_KP_REMOTE,
    TCP_OFFSET,
    EE_BODY_NAME,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
)

class RemoteRobotNode(Node):
    def __init__(self):
        super().__init__('remote_robot_node')
        
        self.n_joints_ = N_JOINTS
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.tcp_offset_ = TCP_OFFSET
        
        # PD Gains
        self.kp_ = DEFAULT_KP_REMOTE 
        self.kd_ = DEFAULT_KD_REMOTE
        
        self.torque_limits_ = TORQUE_LIMITS
        self.max_rl_torque_ = MAX_TORQUE_COMPENSATION
        self.joint_names_ = [f'panda_joint{i+1}' for i in range(self.n_joints_)]
        self.initial_joint_config_ = INITIAL_JOINT_CONFIG
        self.ee_body_name_ = EE_BODY_NAME
        
        self.current_q_ = self.initial_joint_config_.copy()
        self.current_qd_ = np.zeros(self.n_joints_, dtype=np.float32)
        self.target_q_ = INITIAL_JOINT_CONFIG.copy()
        self.target_qd_ = np.zeros(self.n_joints_)
        self.current_tau_rl_ = np.zeros(self.n_joints_)
        self.current_local_q_ = INITIAL_JOINT_CONFIG.copy()
        
        # Watchdog & Buffer
        self.last_tau_rl_time_ = 0.0
        self.last_valid_torque_command_ = np.zeros(self.n_joints_)
        
        # MuJoCo
        self.mj_model_ = mujoco.MjModel.from_xml_path(DEFAULT_MUJOCO_MODEL_PATH)
        self.mj_data_ = mujoco.MjData(self.mj_model_)
        
        # Config
        self.declare_parameter('experiment_config', ExperimentConfig.LOW_DELAY.value)
        self.declare_parameter('seed', 50)
        self.config_int_ = self.get_parameter('experiment_config').value
        self.seed_ = self.get_parameter('seed').value
        
        self.delay_config_ = ExperimentConfig(self.config_int_)
        self.action_delay_simulator_ = DelaySimulator(self.control_freq_, self.delay_config_, self.seed_)
        self.torque_command_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        
        # Flags
        self.robot_state_ready_ = False
        self.target_command_ready_ = False
        self.tau_rl_ready_ = False
        self.local_state_ready_ = False

        # --- MULTI-THREADING SETUP ---
        self.cb_group = ReentrantCallbackGroup()
        
        # Subscribers
        self.create_subscription(JointState, 'agent/predict_target', self.target_command_callback, 10, callback_group=self.cb_group)
        self.create_subscription(Float64MultiArray, 'agent/tau_rl', self.tau_rl_callback, 10, callback_group=self.cb_group)
        self.create_subscription(JointState, 'remote_robot/joint_states', self.robot_state_callback, 10, callback_group=self.cb_group)
        self.create_subscription(JointState, 'local_robot/joint_states', self.local_robot_state_callback, 10, callback_group=self.cb_group)
        
        # --- [FIX] CORRECT TOPIC NAME ---
        # We publish exactly where the controller is listening
        self.torque_command_pub_ = self.create_publisher(
            Float64MultiArray, 
            '/joint_tau/torques_desired', 
            10,
            callback_group=self.cb_group
        )
        self.ee_pose_pub_ = self.create_publisher(PointStamped, 'remote_robot/ee_pose', 10, callback_group=self.cb_group)
        
        # --- TIMERS ---
        self.control_timer_ = self.create_timer(self.dt_, self.control_loop_callback, callback_group=self.cb_group)
        self.publish_timer_ = self.create_timer(0.002, self.high_freq_publish_callback, callback_group=self.cb_group)
        
        self.heartbeat_counter_ = 0
        self.get_logger().info(f"REAL ROBOT NODE STARTED. Target: /joint_tau/torques_desired")

    def high_freq_publish_callback(self):
        msg = Float64MultiArray()
        msg.data = self.last_valid_torque_command_.tolist()
        self.torque_command_pub_.publish(msg)

    def target_command_callback(self, msg: JointState) -> None:
        try:
            name_map = {name: i for i, name in enumerate(msg.name)}
            self.target_q_ = np.array([msg.position[name_map[n]] for n in self.joint_names_])
            self.target_qd_ = np.array([msg.velocity[name_map[n]] for n in self.joint_names_])
            if not self.target_command_ready_:
                self.target_command_ready_ = True
        except Exception: pass

    def tau_rl_callback(self, msg: Float64MultiArray) -> None:
        self.current_tau_rl_ = np.array(msg.data, dtype=np.float32)
        self.last_tau_rl_time_ = self.get_clock().now().nanoseconds / 1e9
        if not self.tau_rl_ready_:
            self.tau_rl_ready_ = True

    def robot_state_callback(self, msg: JointState) -> None:
        try:
            name_map = {name: i for i, name in enumerate(msg.name)}
            self.current_q_ = np.array([msg.position[name_map[n]] for n in self.joint_names_])
            self.current_qd_ = np.array([msg.velocity[name_map[n]] for n in self.joint_names_])
            if not self.robot_state_ready_:
                self.robot_state_ready_ = True
                self.get_logger().info("Connected to Real Robot Hardware.")
        except Exception: pass
            
    def local_robot_state_callback(self, msg: JointState) -> None:
        try:
            name_map = {name: i for i, name in enumerate(msg.name)}
            self.current_local_q_ = np.array([msg.position[name_map[n]] for n in self.joint_names_])
            if not self.local_state_ready_:
                self.local_state_ready_ = True
        except Exception: pass
        
    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _get_inverse_dynamics(self, q, v, a_des):
        self.mj_data_.qpos[:self.n_joints_] = q
        self.mj_data_.qvel[:self.n_joints_] = v
        self.mj_data_.qacc[:self.n_joints_] = a_des
        mujoco.mj_inverse(self.mj_model_, self.mj_data_)
        return self.mj_data_.qfrc_inverse[:self.n_joints_].copy()

    def _get_ee_position(self, q):
        self.mj_data_.qpos[:self.n_joints_] = q
        mujoco.mj_kinematics(self.mj_model_, self.mj_data_)
        return self.mj_data_.body(self.ee_body_name_).xpos.copy()

    def control_loop_callback(self) -> None:
        if not self.robot_state_ready_:
            if self.heartbeat_counter_ % 200 == 0:
                self.get_logger().info("Waiting for hardware connection...")
            self.heartbeat_counter_ += 1
            return

        # Watchdog
        now = self.get_clock().now().nanoseconds / 1e9
        if (now - self.last_tau_rl_time_) > 0.2:
            if self.tau_rl_ready_: 
                 self.get_logger().warn("Watchdog: Agent silent. Zeroing RL torque.", throttle_duration_sec=1)
            self.current_tau_rl_ = np.zeros(self.n_joints_)

        q_target = self.target_q_ 
        qd_target = self.target_qd_ 
        q_current = self.current_q_
        qd_current = self.current_qd_
        
        # 1. PD Control
        q_error = self._normalize_angle(q_target - q_current)
        qd_error = qd_target - qd_current
        acc_desired = self.kp_ * q_error + self.kd_ * qd_error
        
        # 2. Inverse Dynamics
        tau_id = self._get_inverse_dynamics(q_current, qd_current, acc_desired)
        
        # 3. RL Compensation (Clipped)
        tau_rl = np.clip(self.current_tau_rl_, -self.max_rl_torque_, self.max_rl_torque_)
        
        # Combine
        tau_command = tau_id + tau_rl * 0
        
        # 4. Safety Clip & Zero Gripper
        tau_command[-1] = 0.0 
        tau_clipped = np.clip(tau_command, -self.torque_limits_, self.torque_limits_)

        # 5. Delay Simulation
        self.torque_command_history_.append(tau_clipped)
        delay_steps = self.action_delay_simulator_.get_action_delay_steps()
        
        if delay_steps >= len(self.torque_command_history_):
            torque_next = np.zeros(self.n_joints_)
        else:
            torque_next = self.torque_command_history_[-1 - delay_steps]
            
        # Update Buffer
        self.last_valid_torque_command_ = torque_next
        
        ee_pos = self._get_ee_position(q_current)
        ee_msg = PointStamped()
        ee_msg.header.stamp = self.get_clock().now().to_msg()
        ee_msg.header.frame_id = "world" 
        ee_msg.point.x, ee_msg.point.y, ee_msg.point.z = float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])
        self.ee_pose_pub_.publish(ee_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RemoteRobotNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()