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
import time

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
    TRAJECTORY_FREQUENCY,
    DELAY_INPUT_NORM_FACTOR,
    TARGET_DELTA_SCALE,
    WARM_UP_DURATION
)

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Frequency Management
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.last_local_update_time_ = 0.0
        
        # Experiment config setup
        self.default_experiment_config_ = ExperimentConfig.HIGH_DELAY.value
        self.declare_parameter('experiment_config', self.default_experiment_config_)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        
        # Load Models
        self.sac_model_path_ = RL_MODEL_PATH
        self.lstm_model_path_ = LSTM_MODEL_PATH
        
        self.state_estimator_ = StateEstimator().to(self.device_)
        sac_state_dim = (N_JOINTS * 2) + (N_JOINTS * 2 + 1) 
        
        self.actor_ = Actor(state_dim=sac_state_dim).to(self.device_)
        
        self._load_models()

        # Initialize Delay Simulator
        self.delay_simulator_ = DelaySimulator(
            control_freq=self.control_freq_,
            config=self.delay_config_,
            seed=50
        )
        
        # Initialize Buffers
        self.leader_q_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)

        self.current_remote_q_ = np.zeros(N_JOINTS, dtype=np.float32)
        self.current_remote_qd_ = np.zeros(N_JOINTS, dtype=np.float32)
        
        # Warmup Logic
        self.warmup_time_ = WARM_UP_DURATION # seconds
        self.warmup_steps_ = int(self.warmup_time_ * self.control_freq_)
        self.warmup_steps_count_ = 0
        self.buffer_flushed_ = False
        
        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        self.rnn_seq_len_ = RNN_SEQUENCE_LENGTH
        
        # [FIX 1] Pre-fill with CORRECT Initial Config
        self._prefill_buffers()
        
        self.is_leader_ready_ = False
        self.is_remote_ready_ = False
        
        # Publishers & Subscribers
        self.tau_pub_ = self.create_publisher(Float64MultiArray, 'agent/tau_rl', 100)
        self.desired_q_pub_ = self.create_publisher(JointState, 'agent/predict_target', 100)

        self.local_robot_state_subscriber_ = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 10
        )
        self.remote_robot_state_subscriber_ = self.create_subscription(
            JointState, 'remote_robot/joint_states', self.remote_robot_state_callback, 10
        )
        
        self.control_timer_ = self.create_timer(self.dt_, self.control_loop_callback)
        self.get_logger().info("Agent Node initialized.")

    def _load_models(self):
        try:
            lstm_ckpt = torch.load(
                self.lstm_model_path_, 
                map_location=self.device_, 
                weights_only=False
            )
            self.state_estimator_.load_state_dict(lstm_ckpt.get('state_estimator_state_dict', lstm_ckpt))
            self.state_estimator_.eval()
            
            sac_ckpt = torch.load(
                self.sac_model_path_, 
                map_location=self.device_, 
                weights_only=False
            )
            self.actor_.load_state_dict(sac_ckpt['actor_state_dict'])
            self.actor_.eval()
        except Exception as e:
            self.get_logger().fatal(f"Model load failed: {e}")
            raise

    def _prefill_buffers(self) -> None:
        """
        [FIX 1] Pre-fill with INITIAL_JOINT_CONFIG instead of Zeros.
        This ensures the LSTM sees a valid state (Stationary Robot) instead of a discontinuity.
        """
        q_init = INITIAL_JOINT_CONFIG.astype(np.float32)
        qd_init = np.zeros(N_JOINTS, dtype=np.float32)
        
        # We add a bit more than RNN sequence length
        self.num_prefill_ = self.rnn_seq_len_ + 20
        
        for _ in range(self.num_prefill_):
            self.leader_q_history_.append(q_init)
            self.leader_qd_history_.append(qd_init)
            
        self.get_logger().info(f"Prefilled with {self.num_prefill_} INITIAL_JOINT_CONFIG states.")

    def local_robot_state_callback(self, msg: JointState) -> None:
        # Enforce Frequency Matching (approx)
        current_time = self.get_clock().now().nanoseconds / 1e9
        if (current_time - self.last_local_update_time_) < (self.dt_ * 0.95):
            return
        self.last_local_update_time_ = current_time

        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index_map[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index_map[name]] for name in self.target_joint_names_]

            q_new = np.array(pos, dtype=np.float32)
            qd_new = np.array(vel, dtype=np.float32)
            
            self.leader_q_history_.append(q_new)
            self.leader_qd_history_.append(qd_new)
            
            # Only declare ready if we have enough REAL data (or valid prefill)
            if not self.is_leader_ready_ and len(self.leader_q_history_) > self.rnn_seq_len_:
                self.is_leader_ready_ = True
        except (KeyError, IndexError):
            pass

    def remote_robot_state_callback(self, msg: JointState) -> None:
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index_map[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index_map[name]] for name in self.target_joint_names_]
            
            self.current_remote_q_ = np.array(pos, dtype=np.float32)
            self.current_remote_qd_ = np.array(vel, dtype=np.float32)
            self.is_remote_ready_ = True
        except:
            pass
    
    def _get_delayed_leader_data(self) -> tuple[np.ndarray, float]:
        history_len = len(self.leader_q_history_)
        obs_delay_steps = self.delay_simulator_.get_observation_delay_steps(history_len)
        current_delay_scalar = float(obs_delay_steps) / DELAY_INPUT_NORM_FACTOR
        
        most_recent_delayed_idx = -(obs_delay_steps + 1)
        oldest_idx = most_recent_delayed_idx - self.rnn_seq_len_ + 1
        
        buffer_seq = []
        
        for i in range(oldest_idx, most_recent_delayed_idx + 1):
            safe_idx = np.clip(i, -history_len, -1)
            # [FIX] Construct 15D vector: [q, qd, delay]
            step_vector = np.concatenate([
                self.leader_q_history_[safe_idx],
                self.leader_qd_history_[safe_idx],
                [current_delay_scalar] 
            ])
            buffer_seq.append(step_vector)
        
        buffer = np.array(buffer_seq).flatten().astype(np.float32)
        return buffer, current_delay_scalar
        
    def control_loop_callback(self) -> None:
        if not self.is_leader_ready_ or not self.is_remote_ready_:
            return
        
        self.warmup_steps_count_ += 1

        if self.warmup_steps_count_ < self.warmup_steps_:    
            if self.warmup_steps_count_ % 50 == 0:
                 self.get_logger().info(f"STANDBY... {self.warmup_steps_ - self.warmup_steps_count_} steps left")
            
            safe_q = self.current_remote_q_.copy()
            safe_qd = np.zeros(N_JOINTS)
            safe_tau = np.zeros(N_JOINTS)
            self.publish_predicted_target(safe_q, safe_qd)
            self.publish_tau_compensation(safe_tau)
            return
        
        if not self.buffer_flushed_:
            self.buffer_flushed_ = True
            self.get_logger().info("WARMUP COMPLETE! Flushing artificial pre-fill...")
            num_to_pop = min(len(self.leader_q_history_), self.num_prefill_)
            for _ in range(num_to_pop):
                self.leader_q_history_.popleft()
                self.leader_qd_history_.popleft()
            self.get_logger().info(f"Buffer flushed. New size: {len(self.leader_q_history_)}")
        
        try:            
            
            # Get normalized delayed leader data
            raw_delayed_sequence, raw_delay_scalar = self._get_delayed_leader_data()
            normalized_delay_scalar = raw_delay_scalar
            
            # 15D data per step: [q(7), qd(7), delay(1)]
            full_seq_t = torch.tensor(raw_delayed_sequence, dtype=torch.float32).to(self.device_).reshape(1, self.rnn_seq_len_, -1)
            
            # delay tensor (Batch=1, 1)
            delay_t = torch.tensor([[normalized_delay_scalar]], dtype=torch.float32).to(self.device_)
            
            # Remote State (Batch=1, 14)
            raw_remote_state = np.concatenate([self.current_remote_q_, self.current_remote_qd_])
            remote_state_t = torch.tensor(raw_remote_state, dtype=torch.float32).to(self.device_).reshape(1, -1)

            # Local robot state prediction
            with torch.no_grad():
                # Get Scaled Residual from LSTM
                scaled_residual_t, _ = self.state_estimator_(full_seq_t)
                
                # Descale
                predicted_residual_t = scaled_residual_t / TARGET_DELTA_SCALE
                
                # Add to Delayed Observation (The "Anchor")
                # The sequence is [q, qd, delay]. We need just [q, qd] (first 14 dims) from the LAST step.
                last_delayed_obs_t = full_seq_t[:, -1, :N_JOINTS*2] 
                
                # Recover Absolute State
                predicted_state_t = last_delayed_obs_t + predicted_residual_t
                
                # Actor Input: [predicted_state, remote_state, delay]
                actor_input_t = torch.cat([predicted_state_t, remote_state_t, delay_t], dim=1)
                
                action_t, _, _ = self.actor_.sample(actor_input_t, deterministic=True)

            # Publish Results
            predicted_state_np = predicted_state_t.cpu().numpy().flatten()
            predicted_q = predicted_state_np[:N_JOINTS]
            predicted_qd = predicted_state_np[N_JOINTS:] 

            tau_rl = action_t.cpu().numpy().flatten()
            tau_rl[-1] = 0.0 # Safety: No torque on gripper
            
            self.publish_predicted_target(predicted_q, predicted_qd)
            self.publish_tau_compensation(tau_rl)
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def publish_predicted_target(self, q, qd):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.target_joint_names_
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        self.desired_q_pub_.publish(msg)

    def publish_tau_compensation(self, tau):
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.tau_pub_.publish(msg)
    
def main(args=None):
    rclpy.init(args=args)
    agent_node = AgentNode()
    rclpy.spin(agent_node)
    agent_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()