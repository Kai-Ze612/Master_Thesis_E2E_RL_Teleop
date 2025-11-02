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
import os

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Custom imports
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import (
    DelaySimulator,
    ExperimentConfig
)

from Reinforcement_Learning_In_Teleoperation.rl_agent.ppo_policy_network import (
    RecurrentPPOPolicy,
    HiddenStateType
)

from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
    DEFAULT_RL_MODEL_PATH_BASE
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
        self.lstm_hidden_state: HiddenStateType = self.policy.init_hidden_state(
            batch_size=1, device=self.device
        )
        
        # Flag to ensure both robots are ready
        self.local_robot_ready = False
        self.remote_robot_ready = False
        
        # Subscribe to local and remote robot joint states
        self.local_robot_state_subscriber = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 100)

        self.remote_robot_state_subscriber = self.create_subscription(
            JointState, 'remote_robot/joint_states', self.remote_robot_state_callback, 100)

        self.predicted_joint_state_publisher = self.create_publisher(
            JointState, 'agent/command', 100)
        
        self.predicted_tau_compensation_publisher = self.create_publisher(
            JointState, 'agent/tau_compensation', 100)

    def _prefill_buffers(self) -> None:
        """Prefill the history buffers with the initial robot state."""
        q_init = INITIAL_JOINT_CONFIG.copy()
        qd_init = np.zeros(N_JOINTS)
        for _ in range(DEPLOYMENT_HISTORY_BUFFER_SIZE):
            self.leader_q_history.append(q_init)
            self.leader_qd_history.append(qd_init)

        # Set initial remote state
        self.current_remote_q = q_init
        self.current_remote_qd = qd_init
        
    def local_robot_state_callback(self, msg: JointState) -> None:
        """Callback for local robot state updates."""
        self.leader_q_history.append(np.array(msg.position[:N_JOINTS], dtype=np.float32))
        self.leader_qd_history.append(np.array(msg.velocity[:N_JOINTS], dtype=np.float32))
        
        if not self.local_robot_ready:
            self.local_robot_ready = True
            self.get_logger().info("Local robot state received. Local robot is ready.")
        
    def remote_robot_state_callback(self, msg: JointState) -> None:
        """Callback for remote robot state updates."""
        name_to_index_map = {name: i for i, name in enumerate(msg.name)}
        pos = [msg.position[name_to_index_map[name]] for name in self.target_joint_names]
        vel = [msg.velocity[name_to_index_map[name]] for name in self.target_joint_names]
        
        self.current_remote_q_ = np.array(pos, dtype=np.float32)
        self.current_remote_qd_ = np.array(vel, dtype=np.float32)
       
        if not self.is_remote_ready_:
            self.get_logger().info("First REMOTE state received.")
            self.is_remote_ready_ = True
            
    def _get_delayed_leader_state(self) -> np.ndarray:
        """Manually adding delay for incoming leader state."""
        history_len = len(self.leader_q_history)
        
        # Get observation delay
        obs_delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)

        # Get the delayed state from the *end* of the deque
        delay_index = np.clip(obs_delay_steps, 0, history_len - 1)
        delayed_q = self.leader_q_history[-delay_index - 1]
        delayed_qd = self.leader_qd_history[-delay_index - 1]

        # Concatenate to match the RNN's feature_dim (14)
        return np.concatenate([delayed_q, delayed_qd])
    
    def control_loop_callback(self) -> None:
        """Main control loop"""
        """Main loop is running at 500 Hz"""
        
        if not self.is_leader_ready_ or not self.is_remote_ready_:
            if not self.is_leader_ready_:
                self.get_logger().warn("Waiting for leader data...", throttle_duration_sec=5.0)
            if not self.is_remote_ready_:
                self.get_logger().warn("Waiting for remote data...", throttle_duration_sec=5.0)
            return
        
        # Get delayed leader state
        delayed_leader_state = self._get_delayed_leader_state()
        remote_state = np.concatenate([self.current_remote_q_, self.current_remote_qd_])
        
        # Construct observation
        delay_obs_t = torch.tensor(delayed_leader_state, dtype=torch.float32).to(self.device).reshape(1, 1, -1)
        remote_obs_t = torch.tensor(remote_state, dtype=torch.float32).to(self.device).reshape(1, 1, -1)

        with torch.no_grad():
                action_t, _, _, predicted_target_t, new_hidden_state = self.policy_.get_action(
                    delayed_sequence=delay_obs_t,
                    remote_state=remote_obs_t,
                    hidden_state=self.lstm_hidden_state_,
                    deterministic=True
                )
                
        self.lstm_hidden_state_ = new_hidden_state
        
        predicted_q = predicted_target_t.cpu().numpy().flatten()[:N_JOINTS]
        predicted_qd = predicted_target_t.cpu().numpy().flatten()[N_JOINTS:]
        tau_comp = action_t.cpu().numpy().flatten()
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.target_joint_names
        
        # The "position" and "velocity" fields are the agent's *predicted target*
        msg.position = predicted_q.tolist()
        msg.velocity = predicted_qd.tolist()
        
        # The "effort" field is the agent's *torque compensation*
        msg.effort = tau_comp.tolist()
        
        self.command_pub_.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    agent_node = None
    try:
        agent_node = Agent()
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        if agent_node:
            agent_node.get_logger().info("Keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Node failed to initialize or run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if agent_node:
            agent_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()  