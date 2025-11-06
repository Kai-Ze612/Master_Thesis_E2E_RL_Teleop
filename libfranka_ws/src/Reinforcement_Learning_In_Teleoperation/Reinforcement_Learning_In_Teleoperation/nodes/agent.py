"""
Deployment trained RL agent.

In this deployment, the agent node is considered to be on the remote side.

This implies that the communication between the agent and the remote robot is in real-time,
while the communication it receives from the local robot is delayed."

Pipeline:
1. Receive local robot joint states (with delay simulation)
2. Receive remote robot joint states (real-time)
3. Construct observation with delayed local robot history
4. Query RL agent for action
5. Get predicted target and torque compensation
6. Publish desired trajectory point to agent/predict_targret
7. Publish torque compensation to agent/tau_rl
"""

import numpy as np
import torch
from collections import deque
import os

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray 

# Custom imports
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Reinforcement_Learning_In_Teleoperation.rl_agent.ppo_policy_network import (
    RecurrentPPOPolicy, HiddenStateType
)
from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
    DEFAULT_RL_MODEL_PATH_BASE,
    RNN_SEQUENCE_LENGTH
)


class AgentNode(Node):
    """
    Implement RL agent node for trained Agent deployment.
    """

    def __init__(self):
        super().__init__('agent_node')
        
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize control frequency and timer period
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.timer_period_ = 1.0 / self.control_freq_

        # Initialize Experiment Config
        self.default_experiment_config_ = ExperimentConfig.HIGH_DELAY.value
        self.declare_parameter('experiment_config', self.default_experiment_config_)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        
        #################################################################################
        # Only for testing
        self.agent_path_ = os.path.join(DEFAULT_RL_MODEL_PATH_BASE, "config_3_test_1", "final_policy.pth")
        
        # Modification: Use a try...except block for robust loading
        try:
            self.policy_ = RecurrentPPOPolicy.load(self.agent_path_, device=self.device_)
            self.policy_.eval() # Modification: Add .eval() for deployment
            self.get_logger().info(f"Loaded RL agent successfully from {self.agent_path_}")
        except Exception as e:
            self.get_logger().fatal(f"Failed to load RL agent from {self.agent_path_}: {e}")
            raise # Stop the node from starting if the model fails to load
        ###############################################################################################
        
        # # Load agent model path based on experiment config
        # if self.experiment_config_int_ == ExperimentConfig.HIGH_DELAY.value:
        #     self.agent_path_ = os.path.join(DEFAULT_RL_MODEL_PATH_BASE, "config_3", "final_policy.pth")
        # elif self.experiment_config_int_ == ExperimentConfig.MEDIUM_DELAY.value:
        #     self.agent_path_ = os.path.join(DEFAULT_RL_MODEL_PATH_BASE, "config_2", "final_policy.pth")
        # elif self.experiment_config_int_ == ExperimentConfig.LOW_DELAY.value:
        #     self.agent_path_ = os.path.join(DEFAULT_RL_MODEL_PATH_BASE, "config_1", "final_policy.pth")
        # else:
        #     raise ValueError(f"Invalid experiment config: {self.experiment_config_int_}")

        # try:
        #     self.policy_ = RecurrentPPOPolicy.load(self.agent_path_, device=self.device_)
        #     self.policy_.eval()
        #     self.get_logger().info(f"Loaded RL agent successfully from {self.agent_path_}")
        # except Exception as e:
        #     self.get_logger().fatal(f"Failed to load RL agent from {self.agent_path_}: {e}")
        #     raise
        
        # Store the expected sequence length from the loaded policy
        try:
             self.rnn_seq_len_ = self.policy_.seq_length
        except AttributeError:
             self.rnn_seq_len_ = RNN_SEQUENCE_LENGTH  
        self.get_logger().info(f"Using RNN Sequence Length: {self.rnn_seq_len_}")
        
        # Initialize Delay Simulator
        try:
            self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        except ValueError:
            self.get_logger().fatal(f"Invalid 'experiment_config' int: {self.experiment_config_int_}")
            raise

        self.delay_simulator_ = DelaySimulator(
            control_freq=self.control_freq_,
            config=self.delay_config_,
            seed=50 # Fixed seed for experiment reproducibility
        )
        self.get_logger().info(f"Initialized with delay config: {self.delay_config_.name}")
        
        # Initialize Buffers and States
        self.leader_q_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)

        # Initialize current remote state
        self.current_remote_q_ = None
        self.current_remote_qd_ = None
        
        self._prefill_buffers()

        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        
        # Initialize LSTM hidden states
        self.lstm_hidden_state_: HiddenStateType = self.policy_.init_hidden_state(
            batch_size=1, device=self.device_
        )
        
        # Flag of starting both robots
        self.is_leader_ready_ = False
        self.is_remote_ready_ = False
        
        # Consolidate to one publisher, as used in the control loop
        self.tau_pub_ = self.create_publisher(
            Float64MultiArray, 'agent/tau_rl', 100
        )
        
        self.desired_q_pub_ = self.create_publisher(
            JointState, 'agent/predict_target', 100
        )

        self.local_robot_state_subscriber_ = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 10
        )

        self.remote_robot_state_subscriber_ = self.create_subscription(
            JointState, 'remote_robot/joint_states', self.remote_robot_state_callback, 10
        )
        
        self.control_timer_ = self.create_timer(
            self.timer_period_, self.control_loop_callback
        )
        
        self.get_logger().info("Agent Node initialized. Waiting for data from robots...")

    def _prefill_buffers(self) -> None:
        """Prefill the history buffers with the initial robot state."""
        q_init = INITIAL_JOINT_CONFIG.copy()
        qd_init = np.zeros(N_JOINTS)
        for _ in range(DEPLOYMENT_HISTORY_BUFFER_SIZE):
            self.leader_q_history_.append(q_init)
            self.leader_qd_history_.append(qd_init)

        # Set initial remote state
        self.current_remote_q_ = q_init
        self.current_remote_qd_ = qd_init
        
    def local_robot_state_callback(self, msg: JointState) -> None:
        """Callback for local robot state updates."""
        # Re-order joint states to match target_joint_names_
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index_map[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index_map[name]] for name in self.target_joint_names_]

            q_new = np.array(pos, dtype=np.float32)
            qd_new = np.array(vel, dtype=np.float32)
            
            self.leader_q_history_.append(q_new)
            self.leader_qd_history_.append(qd_new)

            # Start flag
            if not self.is_leader_ready_:
                self.is_leader_ready_ = True
                self.get_logger().info("Local robot state received. Local robot is ready.")
        
        except (KeyError, IndexError) as e:
            self.get_logger().warn(
                f"Error re-ordering LOCAL state: {e}. Skipping message."
            )
        
    def remote_robot_state_callback(self, msg: JointState) -> None:
        """Callback for remote robot state updates."""
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index_map[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index_map[name]] for name in self.target_joint_names_]
            
            self.current_remote_q_ = np.array(pos, dtype=np.float32)
            self.current_remote_qd_ = np.array(vel, dtype=np.float32)
            
            if not self.is_remote_ready_:
                self.is_remote_ready_ = True
                self.get_logger().info("First REMOTE state received.")
                
        except (KeyError, IndexError) as e:
            self.get_logger().warn(
                f"Error re-ordering remote state: {e}. Skipping message."
            )
    
    def _get_delayed_leader_sequence(self) -> np.ndarray:
        """Get delayed target sequence for state predictor (LSTM) input."""

        history_len = len(self.leader_q_history_)
        
        # Get current observation delay
        obs_delay_steps = self.delay_simulator_.get_observation_delay_steps(history_len)
        
        # Most recent delayed observation index
        most_recent_delayed_idx = -(obs_delay_steps + 1)
        
        # Oldest index we need
        oldest_idx = most_recent_delayed_idx - self.rnn_seq_len_ + 1
        
        buffer_q = []
        buffer_qd = []
        
        # Iterate from oldest to most recent (FORWARD in time)
        for i in range(oldest_idx, most_recent_delayed_idx + 1):
            # Clip to valid range [-history_len, -1]
            safe_idx = np.clip(i, -history_len, -1)
            buffer_q.append(self.leader_q_history_[safe_idx].copy())
            buffer_qd.append(self.leader_qd_history_[safe_idx].copy())
        
        # Stack and concatenate
        # Create a list of (14,) arrays, then stack into (seq_len, 14)
        sequence = [np.concatenate([q, qd]) for q, qd in zip(buffer_q, buffer_qd)]
        
        return np.array(sequence, dtype=np.float32)
    
    def control_loop_callback(self) -> None:
        """
        Main control loop running at control_freq_
        """
        
        if not self.is_leader_ready_ or not self.is_remote_ready_:
            if not self.is_leader_ready_:
                self.get_logger().warn("Waiting for leader data...", throttle_duration_sec=5.0)
            if not self.is_remote_ready_:
                self.get_logger().warn("Waiting for remote data...", throttle_duration_sec=5.0)
            return
        
        try:
            # Get the full delayed sequence
            delayed_leader_sequence = self._get_delayed_leader_sequence() # Shape (seq_len, 14)
            remote_state = np.concatenate([self.current_remote_q_, self.current_remote_qd_])

            # Construct observation tensor with correct sequence length
            delay_seq_t = torch.tensor(
                delayed_leader_sequence, dtype=torch.float32
            ).to(self.device_).reshape(1, self.rnn_seq_len_, -1) # Shape: (1, seq_len, 14)

            # Construct remote state tensor
            remote_obs_t = torch.tensor(
                remote_state, dtype=torch.float32
            ).to(self.device_).reshape(1, -1) # Shape: (1, 14)

            with torch.no_grad():
                action_t, _, _, predicted_target_t, new_hidden_state = self.policy_.get_action(
                    delayed_sequence=delay_seq_t,
                    remote_state=remote_obs_t,
                    hidden_state=self.lstm_hidden_state_,
                    deterministic=True
                )
                
            self.lstm_hidden_state_ = new_hidden_state
            
            # Extract predictions
            predicted_q = predicted_target_t.cpu().numpy().flatten()[:N_JOINTS]
            predicted_qd = predicted_target_t.cpu().numpy().flatten()[N_JOINTS:]
            tau_rl = action_t.cpu().numpy().flatten()
            
            # Publish predicted target (JointState: position + velocity)
            self.publish_predicted_target(predicted_q, predicted_qd)
            
            # Publish torque compensation (Float64MultiArray: 7D vector)
            self.publish_tau_compensation(tau_rl)
            
            # Optional: Log for debugging
            self.get_logger().info(
                f"Agent output - Predicted q: {np.round(predicted_q, 3)}, "
                f"Tau RL: {np.round(tau_rl, 2)}",
                throttle_duration_sec=2.0
            )
            
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            import traceback
            traceback.print_exc()

    def publish_predicted_target(self, q: np.ndarray, qd: np.ndarray) -> None:
        """Publish predicted target joint state."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.target_joint_names_
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        msg.effort = []  # Empty for predicted targets
        
        self.desired_q_pub_.publish(msg)

    def publish_tau_compensation(self, tau_rl: np.ndarray) -> None:
        """Publish torque compensation as Float64MultiArray."""
        msg = Float64MultiArray()
        msg.data = tau_rl.tolist()
        
        self.tau_pub_.publish(msg)
    
def main(args=None):
    rclpy.init(args=args)
    agent_node = None
    try:
        agent_node = AgentNode()
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
    
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()