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
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Frequency Management
        self.control_freq_ = cfg.DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.last_leader_msg_arrival_time_ = self.get_clock().now().nanoseconds / 1e9
        
        # Experiment config
        self.declare_parameter('experiment_config', ExperimentConfig.LOW_DELAY.value)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        
        # Load Models
        self.state_estimator_ = StateEstimator().to(self.device_)
        self.actor_ = Actor(state_dim=cfg.OBS_DIM).to(self.device_)
        self._load_models()

        # Delay Simulator
        self.delay_simulator_ = DelaySimulator(
            control_freq=self.control_freq_,
            config=self.delay_config_,
            seed=50
        )
        
        # Buffers
        self.leader_q_history_ = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history_ = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.remote_q_history_ = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        self.remote_qd_history_ = deque(maxlen=cfg.REMOTE_HISTORY_LEN)

        self.current_remote_q_ = np.zeros(cfg.N_JOINTS, dtype=np.float32)
        self.current_remote_qd_ = np.zeros(cfg.N_JOINTS, dtype=np.float32)
        
        # Phase Control
        self.warmup_steps_ = int(cfg.WARM_UP_DURATION * self.control_freq_)
        self.warmup_steps_count_ = 0
        self.no_delay_steps_ = int(cfg.NO_DELAY_DURATION * self.control_freq_)
        self.no_delay_steps_count_ = 0
        self.buffer_flushed_ = False
        
        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(cfg.N_JOINTS)]
        self.rnn_seq_len_ = cfg.RNN_SEQUENCE_LENGTH
        self.max_ar_steps_ = cfg.MAX_AR_STEPS 
        
        # Init buffers
        for _ in range(cfg.REMOTE_HISTORY_LEN):
            self.remote_q_history_.append(np.zeros(cfg.N_JOINTS))
            self.remote_qd_history_.append(np.zeros(cfg.N_JOINTS))
        
        self.is_leader_ready_ = False
        self.is_remote_ready_ = False
        
        # [NEW] EMA Filter State
        self.prediction_ema_ = None
        self.ema_alpha_ = cfg.PREDICTION_EMA_ALPHA
        
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
            lstm_ckpt = torch.load(cfg.LSTM_MODEL_PATH, map_location=self.device_)
            if 'state_estimator_state_dict' in lstm_ckpt:
                self.state_estimator_.load_state_dict(lstm_ckpt['state_estimator_state_dict'])
            else:
                self.state_estimator_.load_state_dict(lstm_ckpt)
            self.state_estimator_.eval()
            
            sac_ckpt = torch.load(cfg.RL_MODEL_PATH, map_location=self.device_)
            self.actor_.load_state_dict(sac_ckpt['actor_state_dict'])
            self.actor_.eval()
            self.get_logger().info("Models loaded successfully.")
        except Exception as e:
            self.get_logger().fatal(f"Model load failed: {e}")
            raise

    # --- NORMALIZATION HELPERS ---
    def _normalize_input(self, q, qd, delay_scalar):
        q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
        qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
        return np.concatenate([q_norm, qd_norm, [delay_scalar]])

    def _denormalize_output(self, pred_norm):
        q_norm = pred_norm[:7]
        qd_norm = pred_norm[7:]
        q = (q_norm * cfg.Q_STD) + cfg.Q_MEAN
        qd = (qd_norm * cfg.QD_STD) + cfg.QD_MEAN
        return q, qd
    # -----------------------------

    def local_robot_state_callback(self, msg: JointState) -> None:
        self.last_leader_msg_arrival_time_ = self.get_clock().now().nanoseconds / 1e9
        try:
            name_to_index = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index[name]] for name in self.target_joint_names_]

            q_new = np.array(pos, dtype=np.float32)
            qd_new = np.array(vel, dtype=np.float32)
            
            if not self.is_leader_ready_:
                qd_zero = np.zeros_like(qd_new)
                prefill_count = self.rnn_seq_len_ + 20
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
            name_to_index = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index[name]] for name in self.target_joint_names_]
            self.current_remote_q_ = np.array(pos, dtype=np.float32)
            self.current_remote_qd_ = np.array(vel, dtype=np.float32)
            self.is_remote_ready_ = True
        except:
            pass
    
    def _get_delayed_leader_sequence(self, extra_delay_steps: float = 0.0):
        """Constructs NORMALIZED input sequence for LSTM."""
        history_len = len(self.leader_q_history_)
        obs_delay_steps = self.delay_simulator_.get_observation_delay_steps(history_len)
        
        base_delay_norm = float(obs_delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        extra_delay_norm = extra_delay_steps / cfg.DELAY_INPUT_NORM_FACTOR
        current_delay_scalar = base_delay_norm + extra_delay_norm
        
        most_recent_idx = -(obs_delay_steps + 1)
        start_idx = most_recent_idx - self.rnn_seq_len_ + 1
        
        seq_buffer = []
        for i in range(start_idx, most_recent_idx + 1):
            idx = np.clip(i, -history_len, -1)
            # Normalize every step in sequence
            step_vec = self._normalize_input(
                self.leader_q_history_[idx],
                self.leader_qd_history_[idx],
                current_delay_scalar
            )
            seq_buffer.append(step_vec)
        
        return np.array(seq_buffer), current_delay_scalar, obs_delay_steps

    def _autoregressive_inference(self, initial_seq_t, steps_to_predict, current_delay_scalar):
        """
        Runs LSTM autoregressively to bridge the delay.
        Inputs/Outputs are NORMALIZED tensors.
        """
        with torch.no_grad():
            # 1. Process History
            _, hidden_state = self.state_estimator_.lstm(initial_seq_t)

            # 2. Setup Loop
            curr_input = initial_seq_t[:, -1:, :] # Last frame
            dt_norm = (self.dt_) / cfg.DELAY_INPUT_NORM_FACTOR
            steps_to_run = min(steps_to_predict, self.max_ar_steps_)

            # 3. Autoregressive Loop
            for _ in range(steps_to_run):
                # Predict next state (Normalized)
                # Model handles Euler Integration internally
                pred_state_norm, hidden_state = self.state_estimator_.forward_step(curr_input, hidden_state)
                
                # Update Delay
                current_delay_scalar = max(0.0, current_delay_scalar - dt_norm)
                delay_t = torch.tensor([[[current_delay_scalar]]], device=self.device_)
                
                # Next Input
                curr_input = torch.cat([pred_state_norm, delay_t], dim=2)

            # 4. Final Output (Normalized)
            final_pred_norm = curr_input[0, 0, :14].cpu().numpy()
            return final_pred_norm

    def control_loop_callback(self) -> None:
        if not self.is_leader_ready_ or not self.is_remote_ready_: return
        
        self.warmup_steps_count_ += 1

        # --- PHASE 1: WARMUP ---
        if self.warmup_steps_count_ < self.warmup_steps_:      
            if not hasattr(self, 'initial_remote_q_'):
                self.initial_remote_q_ = self.current_remote_q_.copy()
            
            target_leader_q = self.leader_q_history_[-1]
            alpha = np.clip(self.warmup_steps_count_ / self.warmup_steps_, 0.0, 1.0)
            homing_q = (1.0 - alpha) * self.initial_remote_q_ + alpha * target_leader_q
            
            self.publish_predicted_target(homing_q, np.zeros(cfg.N_JOINTS))
            self.publish_tau_compensation(np.zeros(cfg.N_JOINTS))
            return
        
        if not self.buffer_flushed_:
            self.buffer_flushed_ = True
            num_to_pop = min(len(self.leader_q_history_), self.rnn_seq_len_ + 10) 
            for _ in range(num_to_pop):
                self.leader_q_history_.popleft()
                self.leader_qd_history_.popleft()
        
        try:            
            pred_q = None
            pred_qd = None

            # --- PHASE 2: SEQUENCE FILLING ---
            if self.no_delay_steps_count_ < self.no_delay_steps_:
                self.no_delay_steps_count_ += 1
                pred_q = self.leader_q_history_[-1]
                pred_qd = self.leader_qd_history_[-1]
                normalized_delay_scalar = 0.0

            # --- PHASE 3: DEPLOYMENT ---
            else:
                # Calculate Delay
                curr_time = self.get_clock().now().nanoseconds / 1e9
                time_since_msg = curr_time - self.last_leader_msg_arrival_time_
                staleness_steps = max(0.0, time_since_msg - self.dt_) / self.dt_
                
                # Get Sequence (Normalized)
                seq_norm, norm_delay_scalar, obs_steps = self._get_delayed_leader_sequence(staleness_steps)
                total_steps = int(obs_steps + staleness_steps)
                
                # To Tensor
                seq_t = torch.tensor(seq_norm, dtype=torch.float32).to(self.device_).unsqueeze(0) # (1, Seq, 15)
                
                # Run Inference
                if total_steps > 0:
                    final_pred_norm = self._autoregressive_inference(seq_t, total_steps, norm_delay_scalar)
                    # Denormalize
                    pred_q, pred_qd = self._denormalize_output(final_pred_norm)
                else:
                    # Zero delay case (fallback to last obs)
                    raw_q = self.leader_q_history_[-1]
                    raw_qd = self.leader_qd_history_[-1]
                    pred_q, pred_qd = raw_q, raw_qd

                # Apply EMA Filter
                if self.prediction_ema_ is None:
                    self.prediction_ema_ = pred_q
                else:
                    self.prediction_ema_ = self.ema_alpha_ * pred_q + (1.0 - self.ema_alpha_) * self.prediction_ema_
                
                pred_q = self.prediction_ema_

            # --- SAC POLICY ---
            self.remote_q_history_.append(self.current_remote_q_.copy())
            self.remote_qd_history_.append(self.current_remote_qd_.copy())
            
            rem_q_hist = np.concatenate(list(self.remote_q_history_))
            rem_qd_hist = np.concatenate(list(self.remote_qd_history_))
            
            error_q = pred_q - self.current_remote_q_
            error_qd = pred_qd - self.current_remote_qd_
            
            # Construct Obs (Ensure delay scalar is passed)
            d_scalar = normalized_delay_scalar if 'normalized_delay_scalar' in locals() else 0.0
            
            obs_vec = np.concatenate([
                self.current_remote_q_, self.current_remote_qd_,
                rem_q_hist, rem_qd_hist,
                pred_q, pred_qd,
                error_q, error_qd,
                [d_scalar]
            ]).astype(np.float32)
            
            actor_input_t = torch.tensor(obs_vec, dtype=torch.float32).to(self.device_).unsqueeze(0)
            action_t, _, _ = self.actor_.sample(actor_input_t, deterministic=True)

            tau_rl = action_t.detach().cpu().numpy().flatten()
            tau_rl[-1] = 0.0 
            
            self.publish_predicted_target(pred_q, pred_qd)
            self.publish_tau_compensation(tau_rl)
            
        except Exception as e:
            self.get_logger().error(f"Control Loop Error: {e}")

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