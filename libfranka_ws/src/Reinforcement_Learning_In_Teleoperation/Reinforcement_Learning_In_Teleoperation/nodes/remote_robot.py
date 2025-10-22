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
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory
from collections import deque

# Mujoco imports
import mujoco

# Python imports
import numpy as np
import os
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

# Custom imports
import torch
from stable_baselines3 import SAC
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator
from Reinforcement_Learning_In_Teleoperation.config import (
    N_JOINTS,
    EE_BODY_NAME,
    TCP_OFFSET,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    TORQUE_LIMITS,
    INITIAL_JOINT_CONFIG,
    DEFAULT_CONTROL_FREQ,
    DEFAULT_PUBLISH_FREQ,
    KP_REMOTE_NOMINAL,
    KD_REMOTE_NOMINAL,
    ACTION_HISTORY_LEN,
    TARGET_HISTORY_LEN,
    DEFAULT_MODEL_PATH,
    DEFAULT_RL_MODEL_PATH
)

class RemoteRobot(Node):
    def __init__(self):
        super().__init__('remote_robot_node')
        
        self._init_parameters()
        self._init_mujoco()
        self._init_controllers()
        self._init_compensation_model()
        self._init_ros_interfaces()
        self._init_delay_simulator()
        
    def _init_parameters(self):
        
        self.num_joints: int = N_JOINTS
        self.ee_body_name: str = EE_BODY_NAME
        self.tcp_offset: NDArray[np.float32] = TCP_OFFSET
        self.joint_limits_lower: NDArray[np.float32] = JOINT_LIMITS_LOWER
        self.joint_limits_upper: NDArray[np.float32] = JOINT_LIMITS_UPPER
        self.torque_limits: NDArray[np.float32] = TORQUE_LIMITS
        self.initial_joint_config: NDArray[np.float32] = INITIAL_JOINT_CONFIG
        self.control_freq: float = DEFAULT_CONTROL_FREQ
        self.publish_freq: float = DEFAULT_PUBLISH_FREQ
        self.kp_remote_nominal: NDArray[np.float32] = KP_REMOTE_NOMINAL
        self.kd_remote_nominal: NDArray[np.float32] = KD_REMOTE_NOMINAL
        self.model_path: str = DEFAULT_MODEL_PATH
        self.rl_model_path: str = DEFAULT_RL_MODEL_PATH
        
        # dt
        self.dt = 1.0 / self.control_freq
        
        # Real robot states
        self.real_joint_positions = self.initial_qpos.copy()
        self.real_joint_velocities = np.zeros(self.num_joints)
        
        # Connection monitoring
        self.robot_connected = False
        
        # RL compensation toggle
        self.action_history: deque[NDArray[np.float32]] = deque(maxlen=ACTION_HISTORY_LEN)
        self.target_history: deque[NDArray[np.float32]] = deque(maxlen=TARGET_HISTORY_LEN)
        
        # History buffer for RL observation
        self.target_q_history = deque(maxlen=self.target_history_len)
        self.target_qd_history = deque(maxlen=self.target_history_len)
        self.action_history = deque(maxlen=self.action_history_len)
        
        # Initialize histories
        for _ in range(self.target_history_len):
            self.target_q_history.append(self.initial_qpos.copy())
            self.target_qd_history.append(np.zeros(self.num_joints))
            
        for _ in range(self.action_history_len):
            # RL action is typically [-1, 1], so init history with 0s
            self.action_history.append(np.zeros(self.num_joints))
            
    def _init_mujoco(self):
        """ Initialize Mujoco model for Inverse Dynamics Computation."""
        self.mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.get_logger().info("Mujoco model loaded for Inverse Dynamics computation initialized.")
    
    def _init_compensation_model(self):
        """ Load trained RL model for torque compensation."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compensation_model = SAC.load(self.rl_model_path, device=self.device)
        
        self.get_logger().info("RL compensation model loaded.")
    
    def _init_ros_interfaces(self):
        """ Initialize ROS2 publishers and subscribers."""
        # Subscribers
        self.create_subscription(
            JointState,
            '/real_robot/joint_states',
            self.real_robot_joint_state_callback,
            10
        )
        
        self.create_subscription(
            JointState,
            '/desired_robot/joint_states',
            self.desired_robot_joint_state_callback,
            10
        )
        
        # Publishers
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray,
            '/real_robot/joint_commands',
            10
        )
        
        self.get_logger().info("ROS2 interfaces initialized.")
                
