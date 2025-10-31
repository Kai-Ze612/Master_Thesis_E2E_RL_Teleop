"""
Deployment trained RL agent.

In this deployment, the agent node is considered to be on the remote side.
This implies that the communication between the agent and the remote robot is in real-time,
while the communication it receives from the local robot is delayed."
"""

# Python imports
import numpy as np
import torch
from collections import deque
import sys
from typing import Tuple, Optional
import os

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Custom imports
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Reinforcement_Learning_In_Teleoperation.rl_agent.ppo_policy_network import (RecurrentPPOPolicy, HiddenStateType)
from Reinforcement_Learning_In_Teleoperation.config.robot_config import(
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
    DEFAULT_RL_MODEL_PATH_BASE,
)

class Agent(Node):
    """
    Implement RL agent node for trained Agent deployment.
    """

    def __init__(self):
        super().__init__('agent_node')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.control_freq = DEFAULT_CONTROL_FREQ
        self.timer_period = 1.0 / self.control_freq
        
        # Initialize Experiment Config
        self.default_experiment_config = ExperimentConfig.HIGH_DELAY.value
        self.declare_parameter('experiment_config', self.default_experiment_config)
        self.experiment_config = self.get_parameter('experiment_config').value

        # Load agent model path based on experiment config        
        if self.experiment_config == 3:
            self.agent_path = os.path.join(DEFAULT_RL_MODEL_PATH_BASE, "config_3", "final_policy.pth")
        elif self.experiment_config == 2:
            self.agent_path = os.path.join(DEFAULT_RL_MODEL_PATH_BASE, "config_2", "final_policy.pth")
        elif self.experiment_config == 1:
            self.agent_path = os.path.join(DEFAULT_RL_MODEL_PATH_BASE, "config_1", "final_policy.pth")
        else:
            raise ValueError(f"Invalid experiment config: {self.experiment_config}")                

        try:
            self.policy = RecurrentPPOPolicy.load(self.agent_path, device=self.device)
            self.policy.eval()
            self.get_logger().info(f"Loaded RL agent successfully from {self.agent_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load RL agent from {self.agent_path}: {e}")
        
        # Initialize Delay Simulator    
        self.delay_config = ExperimentConfig(self.experiment_config)
        self.delay_simulator = DelaySimulator(
            control_freq=self.control_freq,
            config=self.delay_config,
            seed=50 #  Fixed seed for experiment reproducibility
        )
        
        # Initialize Buffers and States
        self.leader_q_history = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)

        self.current_remote_q = None
        self.current_remote_qd = None
        
        # Call prefill buffers function
        self._prefill_buffers()

        self.target_joint_names = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        
        # Initialize LSTM hidden states
        self.lstm_hidden_states: HiddenStateType = self.policy.init_hidden_states(batch_size=1, device=self.device)
        
        # Flag to ensure both robots are ready
        self.local_robot_ready = False
        self.remote_robot_ready = False
        
        # Subscribe to local and remote robot joint states
        self.local_robot_state_subscriber = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 100)

        self.remote_robot_state_subscriber = self.create_subscription(
            JointState, 'remote_robot/joint_states', self.remote_robot_state_callback, 100)
        
        self.command_publisher = self.create_publisher(
            JointState, 'agent/command', 10)    
    
    
    
    
    def local_robot_state_callback(self, msg: JointState) -> None:
    
    
    def remote_robot_state_callback(self, msg: JointState) -> None:
        pass    