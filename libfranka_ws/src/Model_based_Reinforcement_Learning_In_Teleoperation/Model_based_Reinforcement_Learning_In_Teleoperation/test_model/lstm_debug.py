import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import torch
from collections import deque
import os

from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
    RNN_SEQUENCE_LENGTH,
    LSTM_MODEL_PATH,
    TRAJECTORY_FREQUENCY,
    INITIAL_JOINT_CONFIG,
)

class LSTMTestNode(Node):
    def __init__(self):
        super().__init__('lstm_test_node')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.control_freq = DEFAULT_CONTROL_FREQ
        self.timer_period = 1.0 / self.control_freq
        self.dt = self.timer_period
        self.last_update_time = 0.0
        
        self.delay_config = ExperimentConfig.LOW_DELAY
        self.delay_simulator = DelaySimulator(self.control_freq, self.delay_config, seed=50)
        
        self.state_estimator = StateEstimator().to(self.device)
        if not os.path.exists(LSTM_MODEL_PATH):
            raise FileNotFoundError(LSTM_MODEL_PATH)
        
        checkpoint = torch.load(LSTM_MODEL_PATH, map_location=self.device)
        key = 'state_estimator_state_dict' if 'state_estimator_state_dict' in checkpoint else None
        self.state_estimator.load_state_dict(checkpoint[key] if key else checkpoint)
        self.state_estimator.eval()
        
        self.leader_q_history = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        
        # Configured sequence length (Max capacity)
        self.config_seq_len = RNN_SEQUENCE_LENGTH 
        self.target_joint_names = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        
        self._prefill_buffers()
        
        self.warmup_time = 1.0 / TRAJECTORY_FREQUENCY
        self.warmup_steps = int(self.warmup_time * self.control_freq)
        self.warmup_steps_count = 0
        self.buffer_flushed = False
        
        self.latest_q = INITIAL_JOINT_CONFIG.copy().astype(np.float32)
        self.latest_qd = np.zeros(N_JOINTS, dtype=np.float32)
        
        self.local_sub = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_callback, 10
        )
        self.timer = self.create_timer(self.timer_period, self.predict_callback)
        
        self.get_logger().info(f"LSTM Test Node Ready. Model trained Seq Len: {self.config_seq_len}")
    
    def _prefill_buffers(self):
        q_init = INITIAL_JOINT_CONFIG.astype(np.float32)
        qd_init = np.zeros(N_JOINTS, dtype=np.float32)
        self.num_prefill = self.config_seq_len + 20
        for _ in range(self.num_prefill):
            self.leader_q_history.append(q_init)
            self.leader_qd_history.append(qd_init)

    def local_callback(self, msg: JointState):
        current_time = self.get_clock().now().nanoseconds / 1e9
        if (current_time - self.last_update_time) < (self.dt * 0.95):
            return
        self.last_update_time = current_time

        try:
            name_map = {name: i for i, name in enumerate(msg.name)}
            q = np.array([msg.position[name_map[name]] for name in self.target_joint_names], dtype=np.float32)
            qd = np.array([msg.velocity[name_map[name]] for name in self.target_joint_names], dtype=np.float32)
            
            self.leader_q_history.append(q)
            self.leader_qd_history.append(qd)
            self.latest_q = q.copy()
            self.latest_qd = qd.copy()
        except:
            pass
    
    def _get_delayed_sequence(self) -> np.ndarray:
        history_len = len(self.leader_q_history)
        delay_steps = self.delay_simulator.get_observation_delay_steps(history_len)
        recent_idx = -(delay_steps + 1) 
        oldest_idx = recent_idx - self.config_seq_len + 1
        
        buffer_q = []
        buffer_qd = []
        
        padding_needed = 0
        if abs(oldest_idx) > history_len:
            padding_needed = abs(oldest_idx) - history_len
            q_static = INITIAL_JOINT_CONFIG.astype(np.float32)
            qd_static = np.zeros(N_JOINTS, dtype=np.float32)
            for _ in range(padding_needed):
                buffer_q.append(q_static)
                buffer_qd.append(qd_static)
        
        start_iter = max(oldest_idx, -history_len)
        for i in range(start_iter, recent_idx + 1):
            buffer_q.append(self.leader_q_history[i].copy())
            buffer_qd.append(self.leader_qd_history[i].copy())
        
        seq = np.stack([np.concatenate([q, qd]) for q, qd in zip(buffer_q, buffer_qd)])
        return seq.flatten().astype(np.float32)
    
    def predict_callback(self):
        self.warmup_steps_count += 1
        if self.warmup_steps_count < self.warmup_steps:
            return
        
        if not self.buffer_flushed:
            self.buffer_flushed = True
            num_to_pop = min(len(self.leader_q_history), self.num_prefill)
            for _ in range(num_to_pop):
                self.leader_q_history.popleft()
                self.leader_qd_history.popleft()

        try:
            # 1. Get Sequence
            seq_flat = self._get_delayed_sequence()
            seq_t_full = torch.tensor(seq_flat).to(self.device).reshape(1, self.config_seq_len, -1).float()
            
            # === INPUT DATA CHECK ===
            inputs_np = seq_t_full.cpu().numpy().squeeze()
            input_q = inputs_np[:, :N_JOINTS]
            self.get_logger().info(f"\n=== INPUT DATA CHECK (Min/Max) ===")
            self.get_logger().info(f"Pos: {np.min(input_q):.3f} / {np.max(input_q):.3f}")
            
            # 2. Test Candidate Lengths
            # [CHANGE] We test 30 and 50 (Middle ground) against 200 (Full)
            candidates = [30, 50, self.config_seq_len] 
            
            for length in candidates:
                if length > self.config_seq_len: continue
                
                seq_slice = seq_t_full[:, -length:, :] 
                
                with torch.no_grad():
                    pred_t, _ = self.state_estimator(seq_slice)
                
                pred_np = pred_t.cpu().numpy().flatten()
                pred_q = pred_np[:N_JOINTS]
                
                error = np.linalg.norm(pred_q - self.latest_q)
                
                self.get_logger().info(f"--- Len {length:3d} | Err: {error:.4f} | Pred[0]: {pred_q[0]:.3f}")

            self.get_logger().info("=========================================")

        except Exception as e:
            self.get_logger().error(f"Prediction error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LSTMTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()