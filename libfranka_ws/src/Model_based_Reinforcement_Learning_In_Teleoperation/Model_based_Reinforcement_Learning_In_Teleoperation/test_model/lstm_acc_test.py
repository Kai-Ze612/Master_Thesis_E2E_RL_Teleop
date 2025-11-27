import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from typing import Tuple, Optional

# Custom Imports
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
    RNN_SEQUENCE_LENGTH,
    RNN_HIDDEN_DIM,
    LSTM_MODEL_PATH,
    TARGET_DELTA_SCALE, # Scaling factor
    DELAY_INPUT_NORM_FACTOR # Normalization factor for delay input
)

# ==============================================================================
# 1. HARDCODED NETWORK ARCHITECTURE (Matches your Checkpoint)
# ==============================================================================
class StateEstimator(nn.Module):
    def __init__(
        self,
        input_dim_total: int = 15,  # 7 q + 7 qd + 1 delay
        hidden_dim: int = 256,      # Fixed to match typical config
        num_layers: int = 2,        # Fixed to match your checkpoint error log
        output_dim: int = 14,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Core
        self.lstm = nn.LSTM(
            input_size=input_dim_total,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Head (named 'fc' to match checkpoint)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        residual = self.fc(last_hidden)
        return residual, last_hidden

    def forward_step(
        self, 
        x_step: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single step forward for autoregressive loop"""
        lstm_out, new_hidden = self.lstm(x_step, hidden_state)
        residual = self.fc(lstm_out[:, -1, :])
        return residual, new_hidden

# ==============================================================================
# 2. TEST NODE
# ==============================================================================
class LSTMTestNode(Node):
    def __init__(self):
        super().__init__('lstm_test_node')
        
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.last_local_update_time_ = 0.0
        self.test_step_counter_ = 0  # <--- Added Step Counter
        
        # Delay Config
        self.declare_parameter('experiment_config', ExperimentConfig.FULL_RANGE_COVER.value)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        
        # 1. Initialize Model with CORRECT Dimensions
        self.state_estimator_ = StateEstimator(
            input_dim_total=15, 
            hidden_dim=RNN_HIDDEN_DIM, 
            num_layers=2 
        ).to(self.device_)
        
        self._load_model()

        # Simulator
        self.delay_simulator_ = DelaySimulator(
            control_freq=self.control_freq_,
            config=self.delay_config_,
            seed=100
        )
        
        # Buffers
        self.leader_q_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        self.rnn_seq_len_ = RNN_SEQUENCE_LENGTH
        
        # Pre-fill buffers
        self._prefill_buffers()
        self.is_leader_ready_ = False
        
        # Communication
        self.local_robot_state_subscriber_ = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 10
        )
        self.control_timer_ = self.create_timer(self.dt_, self.test_loop_callback)
        
        self.get_logger().info("LSTM Test Node initialized with Embedded Architecture.")

    def _load_model(self):
        try:
            lstm_ckpt = torch.load(LSTM_MODEL_PATH, map_location=self.device_)
            
            if isinstance(lstm_ckpt, dict) and 'state_estimator_state_dict' in lstm_ckpt:
                state_dict = lstm_ckpt['state_estimator_state_dict']
            else:
                state_dict = lstm_ckpt

            self.state_estimator_.load_state_dict(state_dict)
            self.state_estimator_.eval()
            self.get_logger().info(f"LSTM loaded successfully.")
        except Exception as e:
            self.get_logger().fatal(f"Model load failed: {e}")
            raise

    def _prefill_buffers(self):
        q_init = INITIAL_JOINT_CONFIG.astype(np.float32)
        qd_init = np.zeros(N_JOINTS, dtype=np.float32)
        for _ in range(self.rnn_seq_len_ + 50):
            self.leader_q_history_.append(q_init)
            self.leader_qd_history_.append(qd_init)

    def local_robot_state_callback(self, msg: JointState):
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
            self.is_leader_ready_ = True
        except Exception:
            pass

    def test_loop_callback(self):
        if not self.is_leader_ready_:
            return
        
        self.test_step_counter_ += 1  # Increment step counter

        try:
            history_len = len(self.leader_q_history_)
            
            # 1. Determine Delay
            delay_steps = int(self.delay_simulator_.get_observation_delay_steps(history_len))
            if delay_steps == 0:
                return 

            # 2. Prepare History Context
            delayed_idx = -1 - delay_steps
            seq_buffer = []
            start_idx = delayed_idx - self.rnn_seq_len_ + 1
            norm_delay_input = float(delay_steps) / DELAY_INPUT_NORM_FACTOR

            for i in range(start_idx, delayed_idx + 1):
                idx = max(-history_len, i)
                step_vec = np.concatenate([
                    self.leader_q_history_[idx],
                    self.leader_qd_history_[idx],
                    [norm_delay_input]
                ])
                seq_buffer.append(step_vec)

            input_tensor = torch.tensor(np.array(seq_buffer), dtype=torch.float32).unsqueeze(0).to(self.device_)

            # 3. Autoregressive Rollout
            curr_q = torch.tensor(self.leader_q_history_[delayed_idx], dtype=torch.float32).to(self.device_)
            curr_qd = torch.tensor(self.leader_qd_history_[delayed_idx], dtype=torch.float32).to(self.device_)
            curr_delay_val = norm_delay_input
            dt_norm = (1.0 / self.control_freq_) / DELAY_INPUT_NORM_FACTOR

            with torch.no_grad():
                # A. Warmup
                _, hidden_state = self.state_estimator_.lstm(input_tensor)
                
                # B. Prediction Loop
                for _ in range(delay_steps):
                    step_input = torch.cat([
                        curr_q.view(1,1,-1), 
                        curr_qd.view(1,1,-1), 
                        torch.tensor([curr_delay_val], device=self.device_).view(1,1,1)
                    ], dim=2)

                    residual_t, hidden_state = self.state_estimator_.forward_step(step_input, hidden_state)
                    
                    residual_physical = residual_t[0] * TARGET_DELTA_SCALE
                    residual_physical = torch.clamp(residual_physical, -0.1, 0.1)
                    
                    pred_delta_q = residual_physical[:7]
                    pred_delta_qd = residual_physical[7:]
                    
                    curr_q = curr_q + pred_delta_q
                    curr_qd = curr_qd + pred_delta_qd
                    curr_delay_val += dt_norm

            # 4. Final Result
            predicted_pos = curr_q.cpu().numpy()
            predicted_vel = curr_qd.cpu().numpy()
            
            gt_pos = self.leader_q_history_[-1]
            gt_vel = self.leader_qd_history_[-1]
            
            pos_error = np.linalg.norm(gt_pos - predicted_pos)
            vel_error = np.linalg.norm(gt_vel - predicted_vel)

            status = "OK" if pos_error < 1.0 else "HIGH ERROR"
            
            # --- MODIFICATION START: Print detailed info every 10 steps ---
            if self.test_step_counter_ % 10 == 0:
                np.set_printoptions(precision=4, suppress=True, linewidth=200)
                print(f"\n[Test Step {self.test_step_counter_}] Delay: {delay_steps} steps")
                print(f"  True Q:      {gt_pos}")
                print(f"  Predicted Q: {predicted_pos}")
                print(f"  Error Norm:  {pos_error:.5f}")
                print("-" * 60)
            # --- MODIFICATION END ---

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LSTMTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()