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
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator, Actor
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
    RNN_SEQUENCE_LENGTH,
    RL_MODEL_PATH,
    LSTM_MODEL_PATH,
    DELAY_INPUT_NORM_FACTOR,
    TARGET_DELTA_SCALE,
    WARM_UP_DURATION,
    NO_DELAY_DURATION,
    REMOTE_HISTORY_LEN,
    OBS_DIM,
    INITIAL_JOINT_CONFIG,
)


class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Frequency Management
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.last_leader_msg_arrival_time_ = self.get_clock().now().nanoseconds / 1e9
        
        # Unified Delay Configuration
        self.default_experiment_config_ = ExperimentConfig.HIGH_DELAY.value
        self.declare_parameter('experiment_config', self.default_experiment_config_)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        
        # Seed parameter
        self.declare_parameter('seed', 50)
        self.seed_ = self.get_parameter('seed').value
        
        # Load Models
        self.sac_model_path_ = RL_MODEL_PATH
        self.lstm_model_path_ = LSTM_MODEL_PATH
        
        self.state_estimator_ = StateEstimator().to(self.device_)
        self.actor_ = Actor(state_dim=OBS_DIM).to(self.device_)
        
        self._load_models()

        # Initialize Delay Simulator
        self.delay_simulator_ = DelaySimulator(
            control_freq=self.control_freq_,
            config=self.delay_config_,
            seed=self.seed_
        )
        
        # Initialize Buffers
        self.leader_q_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)

        # Remote History Buffers
        self.remote_q_history_ = deque(maxlen=REMOTE_HISTORY_LEN)
        self.remote_qd_history_ = deque(maxlen=REMOTE_HISTORY_LEN)

        self.current_remote_q_ = np.zeros(N_JOINTS, dtype=np.float32)
        self.current_remote_qd_ = np.zeros(N_JOINTS, dtype=np.float32)
        
        # Warmup Logic
        self.warmup_time_ = WARM_UP_DURATION 
        self.warmup_steps_ = int(self.warmup_time_ * self.control_freq_)
        self.warmup_steps_count_ = 0
        self.buffer_flushed_ = False
        
        # No_delay Logic
        self.no_delay_time_ = NO_DELAY_DURATION 
        self.no_delay_steps_ = int(self.no_delay_time_ * self.control_freq_)
        self.no_delay_steps_count_ = 0
        
        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        self.rnn_seq_len_ = RNN_SEQUENCE_LENGTH
        
        # Autoregressive Safety Limits
        self.max_ar_steps_ = 50 
        
        # Initialize remote history with initial config
        initial_q = INITIAL_JOINT_CONFIG.copy().astype(np.float32)
        initial_qd = np.zeros(N_JOINTS, dtype=np.float32)
        for _ in range(REMOTE_HISTORY_LEN):
            self.remote_q_history_.append(initial_q.copy())
            self.remote_qd_history_.append(initial_qd.copy())
        
        self.is_leader_ready_ = False
        self.is_remote_ready_ = False
        
        # Publishers & Subscribers
        self.tau_pub_ = self.create_publisher(Float64MultiArray, 'agent/tau_rl', 100)
        self.desired_q_pub_ = self.create_publisher(JointState, 'agent/predict_target', 100)

        self.local_robot_state_subscriber_ = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 100
        )
        self.remote_robot_state_subscriber_ = self.create_subscription(
            JointState, 'remote_robot/joint_states', self.remote_robot_state_callback, 100
        )
        
        self.control_timer_ = self.create_timer(self.dt_, self.control_loop_callback)
        self.get_logger().info(f"Agent Node initialized (Config={self.delay_config_.name}, Seed={self.seed_}).")

    def _load_models(self):
        try:
            lstm_ckpt = torch.load(self.lstm_model_path_, map_location=self.device_, weights_only=False)
            if 'state_estimator_state_dict' in lstm_ckpt:
                self.state_estimator_.load_state_dict(lstm_ckpt['state_estimator_state_dict'])
            else:
                self.state_estimator_.load_state_dict(lstm_ckpt)
            self.state_estimator_.eval()
            
            sac_ckpt = torch.load(self.sac_model_path_, map_location=self.device_, weights_only=False)
            self.actor_.load_state_dict(sac_ckpt['actor_state_dict'])
            self.actor_.eval()
            self.get_logger().info("Models loaded successfully.")
        except Exception as e:
            self.get_logger().fatal(f"Model load failed: {e}")
            raise

    def local_robot_state_callback(self, msg: JointState) -> None:
        self.last_leader_msg_arrival_time_ = self.get_clock().now().nanoseconds / 1e9

        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index_map[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index_map[name]] for name in self.target_joint_names_]

            q_new = np.array(pos, dtype=np.float32)
            qd_new = np.array(vel, dtype=np.float32)
            
            if not self.is_leader_ready_:
                qd_zero = np.zeros_like(qd_new)
                prefill_count = self.rnn_seq_len_ + 20
                self.get_logger().info(f"Initializing leader buffer with first state: {q_new[:3]}...")
                
                for _ in range(prefill_count):
                    self.leader_q_history_.append(q_new.copy())
                    self.leader_qd_history_.append(qd_zero.copy())
                
                self.is_leader_ready_ = True
            else:
                self.leader_q_history_.append(q_new)
                self.leader_qd_history_.append(qd_new)

        except (KeyError, IndexError):
            pass

    def remote_robot_state_callback(self, msg: JointState) -> None:
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index_map[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index_map[name]] for name in self.target_joint_names_]
            
            self.current_remote_q_ = np.array(pos, dtype=np.float32)
            self.current_remote_qd_ = np.array(vel, dtype=np.float32)
            
            # Update remote history
            self.remote_q_history_.append(self.current_remote_q_.copy())
            self.remote_qd_history_.append(self.current_remote_qd_.copy())
            
            self.is_remote_ready_ = True
            
        except (KeyError, IndexError) as e:
            self.get_logger().warn(f"Error processing remote robot state: {e}")

    def _get_delayed_leader_data(self, force_no_delay: bool = False, extra_delay_steps: float = 0.0):
        """
        Get delayed leader data for LSTM input.
        Returns the FULL sequence and the normalized delay value.
        """
        history_len = len(self.leader_q_history_)
        
        if force_no_delay:
            obs_delay_steps = 0
        else:
            obs_delay_steps = self.delay_simulator_.get_observation_delay_steps(history_len)
        
        base_delay_norm = float(obs_delay_steps) / DELAY_INPUT_NORM_FACTOR
        extra_delay_norm = extra_delay_steps / DELAY_INPUT_NORM_FACTOR
        current_delay_scalar = base_delay_norm + extra_delay_norm
        
        # Get history slice based on delay
        most_recent_delayed_idx = -(obs_delay_steps + 1)
        oldest_idx = most_recent_delayed_idx - self.rnn_seq_len_ + 1
        
        buffer_seq = []
        for i in range(oldest_idx, most_recent_delayed_idx + 1):
            safe_idx = np.clip(i, -history_len, -1)
            step_vector = np.concatenate([
                self.leader_q_history_[safe_idx],
                self.leader_qd_history_[safe_idx],
                [current_delay_scalar] 
            ])
            buffer_seq.append(step_vector)
        
        buffer = np.array(buffer_seq).astype(np.float32)  # Shape: (seq_len, 15)
        return buffer, current_delay_scalar, obs_delay_steps

    def _autoregressive_inference_fixed(self, full_sequence: np.ndarray, steps_to_predict: int, normalized_delay: float):
        """
        FIXED Autoregressive inference that matches training procedure EXACTLY.
        
        Training procedure (autoregressive_LSTM.py lines 303-324):
        1. cutoff_idx = RNN_SEQUENCE_LENGTH - packet_loss_steps
        2. safe_history = input_seq[:, :cutoff_idx, :]
        3. _, hidden_state = model.lstm(safe_history)
        4. current_input = safe_history[:, -1:, :]  # Shape (B, 1, 15)
        5. AR loop using current_input tensor directly
        
        Key differences from original deployment:
        - Hidden state built from TRUNCATED sequence (not full)
        - AR loop uses 3D tensor format throughout
        - State update happens on tensor, not separate q/qd variables
        """
        
        steps_to_run = min(steps_to_predict, self.max_ar_steps_)
        
        if steps_to_run <= 0:
            # No prediction needed, return last observation
            return full_sequence[-1, :N_JOINTS], full_sequence[-1, N_JOINTS:2*N_JOINTS]
        
        # Convert to tensor: (1, seq_len, 15)
        input_seq_t = torch.tensor(full_sequence, dtype=torch.float32).unsqueeze(0).to(self.device_)
        
        # ============================================================
        # CRITICAL FIX: Match training's truncation strategy
        # ============================================================
        # In training: cutoff_idx = RNN_SEQUENCE_LENGTH - packet_loss_steps
        # safe_history = input_seq[:, :cutoff_idx, :]
        
        cutoff_idx = self.rnn_seq_len_ - steps_to_run
        
        # Ensure we have at least some context
        cutoff_idx = max(cutoff_idx, 1)
        
        # Truncate sequence (simulate packet loss like training)
        safe_history = input_seq_t[:, :cutoff_idx, :]
        
        # Build hidden state from TRUNCATED sequence
        with torch.no_grad():
            _, hidden_state = self.state_estimator_.lstm(safe_history)
        
        # Start from the END of the truncated sequence
        # Shape: (1, 1, 15) - matching training exactly
        current_input = safe_history[:, -1:, :].clone()
        
        # Normalized time step for delay increment
        dt_norm = self.dt_ / DELAY_INPUT_NORM_FACTOR
        
        # ============================================================
        # AR Loop - matching training structure exactly
        # (autoregressive_LSTM.py lines 312-322)
        # ============================================================
        with torch.no_grad():
            for k in range(steps_to_run):
                # Forward step
                pred_delta, hidden_state = self.state_estimator_.forward_step(current_input, hidden_state)
                
                # Extract current state from input tensor
                last_known_state = current_input[:, :, :14]  # (1, 1, 14)
                
                # Apply scaled delta (matching training line 316)
                predicted_next_state = last_known_state + (pred_delta.unsqueeze(1) * TARGET_DELTA_SCALE)
                
                # Optional: clamp to prevent divergence
                # predicted_next_state = torch.clamp(predicted_next_state, -10.0, 10.0)
                
                # Update delay
                current_delay_norm = current_input[:, :, 14:15]  # (1, 1, 1)
                next_delay_norm = current_delay_norm + dt_norm
                
                # Build next input (matching training line 321)
                current_input = torch.cat([predicted_next_state, next_delay_norm], dim=2)
        
        # Extract final prediction
        final_q = current_input[0, 0, :N_JOINTS].cpu().numpy()
        final_qd = current_input[0, 0, N_JOINTS:2*N_JOINTS].cpu().numpy()
        
        return final_q, final_qd

    def control_loop_callback(self) -> None:
        if not self.is_leader_ready_ or not self.is_remote_ready_:
            return
        
        self.warmup_steps_count_ += 1

        # --- PHASE 1: WARMUP ---
        if self.warmup_steps_count_ < self.warmup_steps_:      
            if self.warmup_steps_count_ % 50 == 0:
                 self.get_logger().info(f"STANDBY... {self.warmup_steps_ - self.warmup_steps_count_} steps left")
            
            safe_q = self.current_remote_q_.copy()
            safe_qd = np.zeros(N_JOINTS)
            safe_tau = np.zeros(N_JOINTS)
            self.publish_predicted_target(safe_q, safe_qd)
            self.publish_tau_compensation(safe_tau)
            return
        
        # --- TRANSITION: BUFFER FLUSH ---
        if not self.buffer_flushed_:
            self.buffer_flushed_ = True
            self.get_logger().info("WARMUP COMPLETE! Flushing artificial pre-fill...")
            num_to_pop = min(len(self.leader_q_history_), self.rnn_seq_len_ + 10) 
            for _ in range(num_to_pop):
                self.leader_q_history_.popleft()
                self.leader_qd_history_.popleft()
            self.get_logger().info(f"Buffer flushed. New size: {len(self.leader_q_history_)}")
        
        # --- PHASE 2: DEPLOYMENT ---
        try:            
            self.no_delay_steps_count_ += 1
            is_in_grace_period = (self.no_delay_steps_count_ < self.no_delay_steps_)
            
            current_time = self.get_clock().now().nanoseconds / 1e9
            time_since_last_msg = current_time - self.last_leader_msg_arrival_time_
            
            # Convert staleness to steps
            packet_staleness_seconds = max(0.0, time_since_last_msg - self.dt_)
            packet_staleness_steps = packet_staleness_seconds / self.dt_
            
            # Get delayed leader data
            full_sequence, normalized_delay_scalar, delay_steps = self._get_delayed_leader_data(
                force_no_delay=is_in_grace_period,
                extra_delay_steps=packet_staleness_steps
            )
            
            # Total steps to predict = delay_steps + staleness
            total_steps_to_predict = int(normalized_delay_scalar * DELAY_INPUT_NORM_FACTOR)
            
            # ============================================================
            # Use FIXED AR inference
            # ============================================================
            if total_steps_to_predict > 0:
                pred_q, pred_qd = self._autoregressive_inference_fixed(
                    full_sequence, 
                    total_steps_to_predict, 
                    normalized_delay_scalar
                )
            else:
                pred_q = full_sequence[-1, :N_JOINTS]
                pred_qd = full_sequence[-1, N_JOINTS:2*N_JOINTS]

            # Build SAC observation
            rem_q_hist = np.concatenate(list(self.remote_q_history_))
            rem_qd_hist = np.concatenate(list(self.remote_qd_history_))
            
            error_q = pred_q - self.current_remote_q_
            error_qd = pred_qd - self.current_remote_qd_
            
            obs_vec = np.concatenate([
                self.current_remote_q_,
                self.current_remote_qd_,
                rem_q_hist,
                rem_qd_hist,
                pred_q,
                pred_qd,
                error_q,
                error_qd,
                [normalized_delay_scalar]
            ]).astype(np.float32)
            
            # Actor Inference
            actor_input_t = torch.tensor(obs_vec, dtype=torch.float32).to(self.device_).unsqueeze(0)
            action_t, _, _ = self.actor_.sample(actor_input_t, deterministic=True)

            # Publish Results
            tau_rl = action_t.detach().cpu().numpy().flatten()
            tau_rl[-1] = 0.0
            
            self.publish_predicted_target(pred_q, pred_qd)
            self.publish_tau_compensation(tau_rl)
            
            # Debug logging
            if self.no_delay_steps_count_ % 100 == 0:
                self.get_logger().info(
                    f"Steps to predict: {total_steps_to_predict}, "
                    f"Pred q[0:3]: {np.round(pred_q[:3], 3)}"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            import traceback
            traceback.print_exc()

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