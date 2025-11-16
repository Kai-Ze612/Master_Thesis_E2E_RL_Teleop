"""
Deployment trained RL agent.

In this deployment, the agent node is considered to be on the remote side.

This implies that the communication between the agent and the remote robot is in real-time,
while the communication it receives from the local robot is delayed."

Pipeline:
1. Receive local robot joint states (delayed) -> Add to leader history buffer.
2. Receive remote robot joint states (real-time) -> Store current remote state.
3. In the control loop:
    a. Get the last delayed observation from the leader history.
    b. Run the LSTM StateEstimator (statefully) to get the predicted_target.
    c. Get the current remote_state.
    d. Concatenate (predicted_target, remote_state) as input for the Actor.
    e. Run the Actor network to get the torque_compensation (action).
4. Publish predicted_target to 'agent/predict_target'.
5. Publish torque_compensation to 'agent/tau_rl'.
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

from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import (
    StateEstimator, Actor
)
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
    RNN_SEQUENCE_LENGTH,
    RL_MODEL_PATH,
    LSTM_MODEL_PATH,
)


class AgentNode(Node):
    """
    Implement RL agent node for trained Agent deployment.
    """

    def __init__(self):
        super().__init__('agent_node')
        
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Control freq parameters
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.timer_period_ = 1.0 / self.control_freq_
        
        # Experiment config
        self.default_experiment_config_ = ExperimentConfig.HIGH_DELAY.value
        self.declare_parameter('experiment_config', self.default_experiment_config_)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        try:
            self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        except ValueError:
            self.get_logger().fatal(f"Invalid 'experiment_config' int: {self.experiment_config_int_}")
            raise
        self.get_logger().info(f"Initialized with delay config: {self.delay_config_.name}")

        # Load LSTM and trained RL
        self.sac_model_path_ = RL_MODEL_PATH
        self.lstm_model_path_ = LSTM_MODEL_PATH

        # Initialize Models
        self.state_estimator_ = StateEstimator().to(self.device_)
        self.actor_ = Actor().to(self.device_)

        # Load State Estimator from PRE-TRAINED file
        if not os.path.exists(self.lstm_model_path_):
            self.get_logger().fatal(f"LSTM model file not found at: {self.lstm_model_path_}")
            raise FileNotFoundError(self.lstm_model_path_)
        
        self.get_logger().info(f"Loading PRE-TRAINED LSTM from: {self.lstm_model_path_}")
        try:
            # The pre-trained file saved the state_dict directly or in a dict
            lstm_checkpoint = torch.load(self.lstm_model_path_, map_location=self.device_)
            if 'state_estimator_state_dict' in lstm_checkpoint:
                self.state_estimator_.load_state_dict(lstm_checkpoint['state_estimator_state_dict'])
            else:
                self.state_estimator_.load_state_dict(lstm_checkpoint) # Assume it's the raw state_dict
            
            self.state_estimator_.eval()
            self.get_logger().info("StateEstimator weights loaded from PRE-TRAINED file.")

        except Exception as e:
            self.get_logger().fatal(f"Failed to load PRE-TRAINED LSTM: {e}")
            raise

        # Load Actor from SAC file
        if not os.path.exists(self.sac_model_path_):
            self.get_logger().fatal(f"SAC model file not found at: {self.sac_model_path_}")
            raise FileNotFoundError(self.sac_model_path_)
            
        self.get_logger().info(f"Loading SAC Actor from: {self.sac_model_path_}")
        try:
            sac_checkpoint = torch.load(self.sac_model_path_, map_location=self.device_, weights_only=False)
            self.actor_.load_state_dict(sac_checkpoint['actor_state_dict'])
            self.actor_.eval()
            self.get_logger().info("Actor weights loaded from SAC checkpoint.")

        except Exception as e:
            self.get_logger().fatal(f"Failed to load ACTOR from SAC checkpoint: {e}")
            raise
        
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
        
        

        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        
        # Initialize LSTM hidden states
        # self.lstm_hidden_state_ = self.state_estimator_.init_hidden_state(
        #     batch_size=1, device=self.device_
        # )
        
        self.rnn_seq_len_ = RNN_SEQUENCE_LENGTH
        self._prefill_buffers()
        
        # Flag of starting both robots
        self.is_leader_ready_ = False
        self.is_remote_ready_ = False
        self.leader_messages_received_ = 0
        
        # Publishers
        self.tau_pub_ = self.create_publisher(
            Float64MultiArray, 'agent/tau_rl', 100
        )
        
        self.desired_q_pub_ = self.create_publisher(
            JointState, 'agent/predict_target', 100
        )

        # Subscribers
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
        num_prefill = max(DEPLOYMENT_HISTORY_BUFFER_SIZE, self.rnn_seq_len_ + 200)
        for _ in range(num_prefill):
            self.leader_q_history_.append(q_init)
            self.leader_qd_history_.append(qd_init)
        self.current_remote_q_ = q_init
        self.current_remote_qd_ = qd_init
        self.get_logger().info(f"Prefilled leader history with {num_prefill} initial states.")
        
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

            # --- THIS IS THE MISSING LINE ---
            self.leader_messages_received_ += 1
            
            # Start flag
            if not self.is_leader_ready_:
                # Wait until the buffer has at least one full sequence
                if self.leader_messages_received_ > self.rnn_seq_len_:
                    self.is_leader_ready_ = True
                    self.get_logger().info(
                        f"Local robot history buffer is full ({self.rnn_seq_len_} steps). Agent is ready."
                    )
        
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
    
    def _get_delayed_leader_observation(self) -> np.ndarray:
        """
        Get delayed target sequence for state predictor (LSTM) input.
        This is for STATELESS inference, matching sac_training_algorithm.py.
        """
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
            safe_idx = np.clip(i, -history_len, -1)
            buffer_q.append(self.leader_q_history_[safe_idx].copy())
            buffer_qd.append(self.leader_qd_history_[safe_idx].copy())
        
        # Stack and concatenate
        sequence = [np.concatenate([q, qd]) for q, qd in zip(buffer_q, buffer_qd)]
        
        # Returns shape (rnn_seq_len, 14) e.g., (256, 14)
        return np.array(sequence, dtype=np.float32)
    
    def control_loop_callback(self) -> None:
        """
        Main control loop running at control_freq_
        """
        
        if not self.is_leader_ready_ or not self.is_remote_ready_:
            return
        
        try:            
            # 1. Get the FULL delayed sequence (Shape: (256, 14))
            delayed_leader_sequence = self._get_delayed_leader_observation()
            
            # 2. Get current remote state (Shape: (14,))
            remote_state = np.concatenate([self.current_remote_q_, self.current_remote_qd_])

            # 3. Convert to Tensors for models
            #    Estimator expects: (batch, seq_len, features) -> (1, 256, 14)
            full_seq_t = torch.tensor(
                delayed_leader_sequence, dtype=torch.float32
            ).to(self.device_).reshape(1, self.rnn_seq_len_, -1) 

            remote_state_t = torch.tensor(
                remote_state, dtype=torch.float32
            ).to(self.device_).reshape(1, -1) # Shape: (1, 14)

            with torch.no_grad():
                predicted_target_t, _ = self.state_estimator_(
                    full_seq_t
                )
                
                actor_input_t = torch.cat([predicted_target_t, remote_state_t], dim=1)
                action_t, _, _ = self.actor_.sample(
                    actor_input_t,
                    deterministic=True
                )

            predicted_target_np = predicted_target_t.cpu().numpy().flatten()
            predicted_q = predicted_target_np[:N_JOINTS]
            predicted_qd = predicted_target_np[N_JOINTS:]
            tau_rl = action_t.cpu().numpy().flatten()
            
            self.publish_predicted_target(predicted_q, predicted_qd)
            self.publish_tau_compensation(tau_rl)
            
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