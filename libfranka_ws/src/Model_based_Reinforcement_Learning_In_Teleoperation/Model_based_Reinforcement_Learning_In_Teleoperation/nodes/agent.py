"""
Agent Node:
1. Subscribes to delayed local (leader) and real-time remote (follower) robot joint states.
    - Delayed leader: for LSTM input
    - Remote: for calculating the real time error
2. Predicts the leader's future state using an LSTM-based StateEstimator to compensate for network delay.
3. Uses an Actor network to calculate a torque compensation action based on the predicted leader state and the current remote state.
4. Publishes the predicted leader state to `agent/predict_target`.
5. Publishes the calculated torque compensation to `agent/tau_rl`.
"""


import numpy as np
import torch
from collections import deque
import time

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
        self.use_fp16_ = (self.device_.type == 'cuda')  ## For lower inference time
        
        # Frequency Management
        self.control_freq_ = cfg.DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.last_leader_msg_arrival_time_ = 0.0  # Counting the leader steps
        
        # Message Counting for Incremental Logic
        self.leader_msgs_received_ = 0
        self.leader_msgs_processed_ = 0
        
        self.declare_parameter('experiment_config', ExperimentConfig.LOW_DELAY.value)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        
        # Load Models
        self.state_estimator_ = StateEstimator().to(self.device_)
        self.actor_ = Actor(state_dim=cfg.OBS_DIM).to(self.device_)
        self._load_models()
        
        if self.use_fp16_:
            self.state_estimator_.half()
            self.actor_.half()
            
        self._warmup_models()

        self.delay_simulator_ = DelaySimulator(self.control_freq_, self.delay_config_, seed=50)
        self.pending_leader_packets_ = deque() 
        
        # Buffers (Input to LSTM)
        self.leader_q_history_ = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history_ = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.remote_q_history_ = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        self.remote_qd_history_ = deque(maxlen=cfg.REMOTE_HISTORY_LEN)

        self.current_remote_q_ = np.zeros(cfg.N_JOINTS, dtype=np.float32)
        self.current_remote_qd_ = np.zeros(cfg.N_JOINTS, dtype=np.float32)
        
        self.warmup_steps_ = int(cfg.WARM_UP_DURATION * self.control_freq_)
        self.warmup_steps_count_ = 0
        self.no_delay_steps_ = int(cfg.NO_DELAY_DURATION * self.control_freq_)
        self.no_delay_steps_count_ = 0
        self.buffer_flushed_ = False
        
        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(cfg.N_JOINTS)]
        self.rnn_seq_len_ = cfg.RNN_SEQUENCE_LENGTH
        
        for _ in range(cfg.REMOTE_HISTORY_LEN):
            self.remote_q_history_.append(np.zeros(cfg.N_JOINTS))
            self.remote_qd_history_.append(np.zeros(cfg.N_JOINTS))
        
        self.is_leader_ready_ = False
        self.is_remote_ready_ = False
        self.prediction_ema_ = None
        self.ema_alpha_ = cfg.PREDICTION_EMA_ALPHA
        
        # Persistent State
        self.internal_hidden_state_ = None
        self.last_autoregressive_output_norm_ = None
        self.current_delay_scalar_ = 0.0
        
        self.tau_pub_ = self.create_publisher(Float64MultiArray, 'agent/tau_rl', 100)
        self.desired_q_pub_ = self.create_publisher(JointState, 'agent/predict_target', 100)

        self.local_robot_state_subscriber_ = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 100
        )
        self.remote_robot_state_subscriber_ = self.create_subscription(
            JointState, 'remote_robot/joint_states', self.remote_robot_state_callback, 100
        )
        
        self.control_timer_ = self.create_timer(self.dt_, self.control_loop_callback)
        self.step_counter_ = 0
        
        np.set_printoptions(precision=3, suppress=True, linewidth=200, floatmode='fixed')
        self.get_logger().info("Agent Node Init. OBSERVATION DELAY BUFFER APPLIED.")

    def _load_models(self):
        """
        Load LSTM model
        """
        try:
            lstm_ckpt = torch.load(cfg.LSTM_MODEL_PATH, map_location=self.device_, weights_only=False)
            if 'state_estimator_state_dict' in lstm_ckpt:
                self.state_estimator_.load_state_dict(lstm_ckpt['state_estimator_state_dict'])
            else:
                self.state_estimator_.load_state_dict(lstm_ckpt)
            self.state_estimator_.eval()
            
            sac_ckpt = torch.load(cfg.RL_MODEL_PATH, map_location=self.device_, weights_only=False)
            self.actor_.load_state_dict(sac_ckpt['actor_state_dict'])
            self.actor_.eval()
        except Exception as e:
            self.get_logger().fatal(f"Model load failed: {e}")
            raise

    def _warmup_models(self):
        self.get_logger().info("Warming up models...")
        dtype = torch.float16 if self.use_fp16_ else torch.float32
        with torch.no_grad():
            dummy_seq = torch.zeros((1, cfg.RNN_SEQUENCE_LENGTH, 15), device=self.device_, dtype=dtype)
            self.state_estimator_.lstm(dummy_seq)
            dummy_step = torch.zeros((1, 1, 15), device=self.device_, dtype=dtype)
            self.state_estimator_.forward_step(dummy_step, None)
            dummy_obs = torch.zeros((1, cfg.OBS_DIM), device=self.device_, dtype=dtype)
            self.actor_.sample(dummy_obs)
        self.get_logger().info("Warmup complete.")

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

    def local_robot_state_callback(self, msg: JointState) -> None:
        """
        Real time leader state
        """
        arrival_time = self.get_clock().now().nanoseconds / 1e9
        self.last_leader_msg_arrival_time_ = arrival_time
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
                    # Prefill goes directly to history (no delay logic for warmup)
                    self.leader_q_history_.append(q_new.copy())
                    self.leader_qd_history_.append(qd_zero.copy())
                self.is_leader_ready_ = True
                self.leader_msgs_received_ = prefill_count 
                self.leader_msgs_processed_ = prefill_count
                self.get_logger().info("Leader stream READY.")
            else:
                self.pending_leader_packets_.append({
                    'q': q_new,
                    'qd': qd_new,
                    't': arrival_time
                })
                
        except Exception: pass

    def remote_robot_state_callback(self, msg: JointState) -> None:
        """
        Read remote robot real time state.
        """
        try:
            name_to_index = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index[name]] for name in self.target_joint_names_]
            self.current_remote_q_ = np.array(pos, dtype=np.float32)
            self.current_remote_qd_ = np.array(vel, dtype=np.float32)
            if not self.is_remote_ready_:
                self.is_remote_ready_ = True
                self.get_logger().info("Remote stream READY.")
        except Exception: pass
    
    def _update_history_from_pending(self):
        """
        Manages the flow of leader joint state packets from a pending queue to the history buffer.
        Packets are moved if their arrival time plus the current observation delay has elapsed.
        """
        if not self.pending_leader_packets_:
            return 0, 0.0
        
        history_len = len(self.leader_q_history_)
        obs_delay_steps = self.delay_simulator_.get_observation_delay_steps(history_len)
        obs_delay_sec = obs_delay_steps * self.dt_
        
        # Normalize for Network Input
        norm_delay_scalar = float(obs_delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        
        now = self.get_clock().now().nanoseconds / 1e9
        new_packets_count = 0
        
        while self.pending_leader_packets_:
            packet = self.pending_leader_packets_[0]
            # If packet is older than delay, it is "observed"
            if (now - packet['t']) >= obs_delay_sec:
                p = self.pending_leader_packets_.popleft()
                self.leader_q_history_.append(p['q'])
                self.leader_qd_history_.append(p['qd'])
                self.leader_msgs_received_ += 1 # Now officially 'received' by Agent logic
                new_packets_count += 1
            else:
                break
                
        return new_packets_count, norm_delay_scalar

    def _process_new_leader_packets(self, new_count, norm_delay_scalar):
        """
        Process newly received leader packets, normalize them, and prepare for LSTM input.
        """
        history_len = len(self.leader_q_history_)
        start_idx = history_len - new_count
        seq_buffer = []
        for i in range(start_idx, history_len):
            step_vec = self._normalize_input(
                self.leader_q_history_[i],
                self.leader_qd_history_[i],
                norm_delay_scalar
            )
            seq_buffer.append(step_vec)
        return np.array(seq_buffer)

    def control_loop_callback(self) -> None:
        """
        State Estimator Logic:
        1. Input 15D data, output 14D (estimate one step)
        2. If because of delay, there is no new incoming leader data, then use autoregressive prediction (Coasting).
        3. If there is new input, we delete the predicted data from sequence, adding new ground truth data.
        
        RL Logic:
        1. Observation space:
            - Current remote robot joint position (7D)
            - Current remote robot joint velocity (7D)
            - History of remote robot joint positions (7D * N)
            - History of remote robot joint velocities (7D * N)
            - Predicted leader joint position (7D)
            - Predicted leader joint velocity (7D)
        2. Output 7D tau compensation action based on the observation.
        3. Goal: min the tracking error between remote robot and true leader.
        """
        
        if not self.is_leader_ready_ or not self.is_remote_ready_: return
        
        self.step_counter_ += 1
        start_time = time.perf_counter()
        self.warmup_steps_count_ += 1
        
        dtype = torch.float16 if self.use_fp16_ else torch.float32
        
        final_q = np.zeros(cfg.N_JOINTS)
        final_qd = np.zeros(cfg.N_JOINTS)
        final_tau = np.zeros(cfg.N_JOINTS)
        phase_name = "UNKNOWN"

        try:
            if self.warmup_steps_count_ < self.warmup_steps_:
                phase_name = "WARMUP"
                if not hasattr(self, 'initial_remote_q_'):
                    self.initial_remote_q_ = self.current_remote_q_.copy()
                target_leader_q = self.leader_q_history_[-1]
                alpha = np.clip(self.warmup_steps_count_ / self.warmup_steps_, 0.0, 1.0)
                final_q = (1.0 - alpha) * self.initial_remote_q_ + alpha * target_leader_q
                
            elif self.no_delay_steps_count_ < self.no_delay_steps_:
                phase_name = "FILLING"
                self.no_delay_steps_count_ += 1
                while self.pending_leader_packets_:
                    p = self.pending_leader_packets_.popleft()
                    self.leader_q_history_.append(p['q'])
                    self.leader_qd_history_.append(p['qd'])
                    self.leader_msgs_received_ += 1
                
                if not self.buffer_flushed_:
                    self.buffer_flushed_ = True
                    self.leader_msgs_processed_ = self.leader_msgs_received_
                    
                final_q = self.leader_q_history_[-1]
                final_qd = self.leader_qd_history_[-1]
                
                if self.internal_hidden_state_ is None:
                    # Initialize hidden state
                    dummy = torch.zeros((1, 1, 15), device=self.device_, dtype=dtype)
                    _, self.internal_hidden_state_ = self.state_estimator_.lstm(dummy)
                
                current_vec = self._normalize_input(final_q, final_qd, 0.0) # Delay 0 during filling
                self.last_autoregressive_output_norm_ = torch.tensor(
                    current_vec[:14], dtype=dtype, device=self.device_
                ).unsqueeze(0) # Shape [1, 14]

            else:
                dt_norm = (self.dt_) / cfg.DELAY_INPUT_NORM_FACTOR
                pred_q_norm = None
                
                new_packets, current_delay_scalar = self._update_history_from_pending()
                
                if new_packets > 0:
                    phase_name = f"FRESH({new_packets})"
                    seq_norm = self._process_new_leader_packets(new_packets, current_delay_scalar)
                    self.leader_msgs_processed_ += new_packets
                    
                    seq_t = torch.tensor(seq_norm, dtype=dtype, device=self.device_).unsqueeze(0)
                    
                    with torch.no_grad():
                        lstm_out, self.internal_hidden_state_ = self.state_estimator_.lstm(
                            seq_t, self.internal_hidden_state_
                        )
                        last_hidden = lstm_out[:, -1, :]
                        velocity_pred = self.state_estimator_.fc(last_hidden)
                        prev_state = seq_t[:, -1, :14]
                        pred_state_norm = prev_state + velocity_pred * self.state_estimator_.dt_scale
                        
                        self.last_autoregressive_output_norm_ = pred_state_norm
                        self.current_delay_scalar_ = current_delay_scalar
                        pred_q_norm = self.last_autoregressive_output_norm_.float().cpu().numpy()[0]
                else:
                    phase_name = "COASTING"
                    self.current_delay_scalar_ = max(0.0, self.current_delay_scalar_ - dt_norm)
                    
                    delay_t = torch.tensor([[[self.current_delay_scalar_]]], dtype=dtype, device=self.device_)
                    
                    state_in = self.last_autoregressive_output_norm_.unsqueeze(1)
                    
                    curr_input = torch.cat([state_in, delay_t], dim=2)
                    
                    with torch.no_grad():
                        pred_state_norm, self.internal_hidden_state_ = self.state_estimator_.forward_step(
                            curr_input, self.internal_hidden_state_
                        )
                        self.last_autoregressive_output_norm_ = pred_state_norm.squeeze(1)
                        pred_q_norm = self.last_autoregressive_output_norm_.float().cpu().numpy()[0]

                final_q, final_qd = self._denormalize_output(pred_q_norm)

                if self.prediction_ema_ is None:
                    self.prediction_ema_ = final_q
                else:
                    self.prediction_ema_ = self.ema_alpha_ * final_q + (1.0 - self.ema_alpha_) * self.prediction_ema_
                final_q = self.prediction_ema_

                self.remote_q_history_.append(self.current_remote_q_.copy())
                self.remote_qd_history_.append(self.current_remote_qd_.copy())
                rem_q_hist = np.concatenate(list(self.remote_q_history_))
                rem_qd_hist = np.concatenate(list(self.remote_qd_history_))
                error_q = final_q - self.current_remote_q_
                error_qd = final_qd - self.current_remote_qd_
                
                obs_vec = np.concatenate([
                    self.current_remote_q_, self.current_remote_qd_,
                    rem_q_hist, rem_qd_hist,
                    final_q, final_qd,
                    error_q, error_qd,
                    [self.current_delay_scalar_]
                ]).astype(np.float32)
                
                actor_input_t = torch.tensor(obs_vec, dtype=dtype, device=self.device_).unsqueeze(0)
                action_t, _, _ = self.actor_.sample(actor_input_t, deterministic=True)
                final_tau = action_t.float().detach().cpu().numpy().flatten()
                final_tau[-1] = 0.0 

            self.publish_predicted_target(final_q, final_qd)
            self.publish_tau_compensation(final_tau)
            
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000.0
            
            if self.step_counter_ % 20 == 0:
                if self.pending_leader_packets_:
                    true_q = self.pending_leader_packets_[-1]['q']
                else:
                    true_q = self.leader_q_history_[-1]
                    
                remote_q = self.current_remote_q_
                pred_error = np.linalg.norm(final_q - true_q)
                track_error = np.linalg.norm(final_q - remote_q)
                
                tq_str = np.array2string(true_q, precision=3, separator=',', suppress_small=True)
                pq_str = np.array2string(final_q, precision=3, separator=',', suppress_small=True)
                rq_str = np.array2string(remote_q, precision=3, separator=',', suppress_small=True)
                
                self.get_logger().info(
                    f"\n[Step {self.step_counter_}] {phase_name} | "
                    f"PredErr: {pred_error:.4f} | "
                    f"TrackErr: {track_error:.4f} | "
                    f"Infer: {inference_time_ms:.2f}ms\n"
                    f"  True Q:   {tq_str}\n"
                    f"  Pred Q:   {pq_str}\n"
                    f"  Remote Q: {rq_str}"
                )
            
        except Exception as e:
            self.get_logger().error(f"Control Loop Error: {e}")
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