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
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator, Actor
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
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
)


class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Frequency Management
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.last_leader_msg_arrival_time_ = self.get_clock().now().nanoseconds / 1e9
        
        # Experiment config
        self.default_experiment_config_ = ExperimentConfig.LOW_DELAY.value
        self.declare_parameter('experiment_config', self.default_experiment_config_)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        
        # Load Models
        self.sac_model_path_ = RL_MODEL_PATH
        self.lstm_model_path_ = LSTM_MODEL_PATH
        
        self.state_estimator_ = StateEstimator().to(self.device_)
        self.actor_ = Actor(state_dim=OBS_DIM).to(self.device_)
        
        self._load_models()

        # Delay Simulator
        self.delay_simulator_ = DelaySimulator(
            control_freq=self.control_freq_,
            config=self.delay_config_,
            seed=50
        )
        
        # Buffers
        self.leader_q_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)

        self.remote_q_history_ = deque(maxlen=REMOTE_HISTORY_LEN)
        self.remote_qd_history_ = deque(maxlen=REMOTE_HISTORY_LEN)

        self.current_remote_q_ = np.zeros(N_JOINTS, dtype=np.float32)
        self.current_remote_qd_ = np.zeros(N_JOINTS, dtype=np.float32)
        
        # --- PHASE CONTROL ---
        self.warmup_time_ = WARM_UP_DURATION 
        self.warmup_steps_ = int(self.warmup_time_ * self.control_freq_)
        self.warmup_steps_count_ = 0
        
        self.no_delay_time_ = NO_DELAY_DURATION 
        self.no_delay_steps_ = int(self.no_delay_time_ * self.control_freq_)
        self.no_delay_steps_count_ = 0
        
        self.buffer_flushed_ = False
        
        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        self.rnn_seq_len_ = RNN_SEQUENCE_LENGTH
        self.max_ar_steps_ = 50 
        
        # Init remote history
        for _ in range(REMOTE_HISTORY_LEN):
            self.remote_q_history_.append(np.zeros(N_JOINTS, dtype=np.float32))
            self.remote_qd_history_.append(np.zeros(N_JOINTS, dtype=np.float32))
        
        self.is_leader_ready_ = False
        self.is_remote_ready_ = False
        
        # Communication
        self.tau_pub_ = self.create_publisher(Float64MultiArray, 'agent/tau_rl', 100)
        self.desired_q_pub_ = self.create_publisher(JointState, 'agent/predict_target', 100)

        self.local_robot_state_subscriber_ = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 100
        )
        self.remote_robot_state_subscriber_ = self.create_subscription(
            JointState, 'remote_robot/joint_states', self.remote_robot_state_callback, 100
        )
        
        self.control_timer_ = self.create_timer(self.dt_, self.control_loop_callback)
        self.get_logger().info("Agent Node initialized.")

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
        """
        BUFFER FILLING LOGIC.
        This runs continuously in the background, ensuring the deque is full
        of valid history (Sequence Filling) for when the LSTM is eventually needed.
        """
        self.last_leader_msg_arrival_time_ = self.get_clock().now().nanoseconds / 1e9
        try:
            name_to_index_map = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index_map[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index_map[name]] for name in self.target_joint_names_]

            q_new = np.array(pos, dtype=np.float32)
            qd_new = np.array(vel, dtype=np.float32)
            
            if not self.is_leader_ready_:
                # Physics fix: Use Zero velocity for artificial stationary history
                qd_zero = np.zeros_like(qd_new)
                prefill_count = self.rnn_seq_len_ + 20
                self.get_logger().info(f"Initializing buffer with Leader State: {q_new[:3]}...")
                for _ in range(prefill_count):
                    self.leader_q_history_.append(q_new.copy())
                    self.leader_qd_history_.append(qd_zero.copy())
                self.is_leader_ready_ = True
            else:
                # Normal Filling
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
            self.is_remote_ready_ = True
        except:
            pass
    
    def _get_delayed_leader_data(self, extra_delay_steps: float = 0.0) -> tuple[np.ndarray, float]:
        """ Fetches input sequence for LSTM based on calculated delay. """
        history_len = len(self.leader_q_history_)
        obs_delay_steps = self.delay_simulator_.get_observation_delay_steps(history_len)
        
        base_delay_norm = float(obs_delay_steps) / DELAY_INPUT_NORM_FACTOR
        extra_delay_norm = extra_delay_steps / DELAY_INPUT_NORM_FACTOR
        current_delay_scalar = base_delay_norm + extra_delay_norm
        
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
        
        buffer = np.array(buffer_seq).flatten().astype(np.float32)
        return buffer, current_delay_scalar

    def _autoregressive_inference(self, initial_seq_t, steps_to_predict, current_delay_scalar):
        with torch.no_grad():
            _, hidden_state = self.state_estimator_.lstm(initial_seq_t)

        last_obs = initial_seq_t[0, -1, :] 
        current_q = last_obs[:N_JOINTS].clone()
        current_qd = last_obs[N_JOINTS:2*N_JOINTS].clone()
        
        dt_norm = self.dt_ / DELAY_INPUT_NORM_FACTOR
        steps_to_run = min(steps_to_predict, self.max_ar_steps_)

        for _ in range(steps_to_run):
            delay_tensor = torch.tensor([current_delay_scalar], device=self.device_)
            step_input = torch.cat([current_q, current_qd, delay_tensor], dim=0).view(1, 1, -1)

            with torch.no_grad():
                scaled_residual_t, hidden_state = self.state_estimator_.forward_step(step_input, hidden_state)

            pred_residual = scaled_residual_t[0] * TARGET_DELTA_SCALE
            pred_residual = torch.clamp(pred_residual, -0.2, 0.2)
            
            current_q = current_q + pred_residual[:N_JOINTS]
            current_qd = current_qd + pred_residual[N_JOINTS:]
            current_delay_scalar += dt_norm

        return current_q, current_qd
        
    def control_loop_callback(self) -> None:
        if not self.is_leader_ready_ or not self.is_remote_ready_:
            return
        
        self.warmup_steps_count_ += 1

        # ==========================================================================
        # PHASE 1: WARMUP & HOMING
        # Goal: Move robot to Leader Start. Buffer fills with Stationary data.
        # LSTM: OFF
        # ==========================================================================
        if self.warmup_steps_count_ < self.warmup_steps_:      
            if self.warmup_steps_count_ % 50 == 0:
                 self.get_logger().info(f"HOMING... {self.warmup_steps_ - self.warmup_steps_count_} steps left")
            
            if not hasattr(self, 'initial_remote_q_'):
                self.initial_remote_q_ = self.current_remote_q_.copy()
                self.get_logger().info(f"Homing Start Position: {self.initial_remote_q_}")

            target_leader_q = self.leader_q_history_[-1]
            alpha = float(self.warmup_steps_count_) / float(self.warmup_steps_)
            alpha = np.clip(alpha, 0.0, 1.0)

            homing_q = (1.0 - alpha) * self.initial_remote_q_ + alpha * target_leader_q
            homing_qd = np.zeros(N_JOINTS) 
            homing_tau = np.zeros(N_JOINTS)

            self.publish_predicted_target(homing_q, homing_qd)
            self.publish_tau_compensation(homing_tau)
            return
        
        # Flush artificial history once warmup is done
        if not self.buffer_flushed_:
            self.buffer_flushed_ = True
            self.get_logger().info("HOMING COMPLETE! Flushing artificial history...")
            num_to_pop = min(len(self.leader_q_history_), self.rnn_seq_len_ + 10) 
            for _ in range(num_to_pop):
                self.leader_q_history_.popleft()
                self.leader_qd_history_.popleft()
            self.get_logger().info("Buffer flushed.")
        
        try:            
            pred_q = None
            pred_qd = None
            normalized_delay_scalar = 0.0

            # ======================================================================
            # PHASE 2: SEQUENCE FILLING (No Delay)
            # Goal: Buffer fills with Movement Data (Sine wave start).
            # LSTM: OFF (Pass-through Leader data directly)
            # ======================================================================
            if self.no_delay_steps_count_ < self.no_delay_steps_:
                self.no_delay_steps_count_ += 1
                
                # Direct Pass-Through
                pred_q = self.leader_q_history_[-1]
                pred_qd = self.leader_qd_history_[-1]
                normalized_delay_scalar = 0.0
                
                if self.no_delay_steps_count_ % 100 == 0:
                    self.get_logger().info("Grace Period: Direct Tracking (Buffer Filling)")

            # ======================================================================
            # PHASE 3: DEPLOYMENT (LSTM Enabled)
            # Goal: Handle Delay.
            # LSTM: ON (Using buffer filled in Phase 2)
            # ======================================================================
            else:
                current_time = self.get_clock().now().nanoseconds / 1e9
                time_since_last_msg = current_time - self.last_leader_msg_arrival_time_
                
                packet_staleness_seconds = max(0.0, time_since_last_msg - self.dt_)
                packet_staleness_steps = packet_staleness_seconds / self.dt_
                
                raw_delayed_sequence, normalized_delay_scalar = self._get_delayed_leader_data(
                    extra_delay_steps=packet_staleness_steps
                )
                
                total_steps_to_predict = int(normalized_delay_scalar * DELAY_INPUT_NORM_FACTOR)
                
                # Run LSTM
                full_seq_t = torch.tensor(raw_delayed_sequence, dtype=torch.float32).to(self.device_).reshape(1, self.rnn_seq_len_, -1)
                
                if total_steps_to_predict > 0:
                    pred_q_t, pred_qd_t = self._autoregressive_inference(
                        full_seq_t, 
                        total_steps_to_predict, 
                        normalized_delay_scalar
                    )
                    pred_q = pred_q_t.cpu().numpy()
                    pred_qd = pred_qd_t.cpu().numpy()
                else:
                    pred_q = full_seq_t[0, -1, :N_JOINTS].cpu().numpy()
                    pred_qd = full_seq_t[0, -1, N_JOINTS:2*N_JOINTS].cpu().numpy()

            # ======================================================================
            # SAC Policy (Always Active for Correction)
            # ======================================================================
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
            
            actor_input_t = torch.tensor(obs_vec, dtype=torch.float32).to(self.device_).unsqueeze(0)
            action_t, _, _ = self.actor_.sample(actor_input_t, deterministic=True)

            tau_rl = action_t.detach().cpu().numpy().flatten()
            tau_rl[-1] = 0.0 # Safety
            
            self.publish_predicted_target(pred_q, pred_qd)
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