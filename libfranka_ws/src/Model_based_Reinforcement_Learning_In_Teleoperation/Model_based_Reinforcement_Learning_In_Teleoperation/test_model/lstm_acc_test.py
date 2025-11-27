import numpy as np
import torch
from collections import deque
import os
import time

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Custom Imports
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
    RNN_SEQUENCE_LENGTH,
    LSTM_MODEL_PATH,
    TRAJECTORY_FREQUENCY,
    TARGET_DELTA_SCALE  # [CRITICAL IMPORT]
)

class LSTMTestNode(Node):
    def __init__(self):
        super().__init__('lstm_test_node')
        
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Frequency Setup
        self.control_freq_ = DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.last_local_update_time_ = 0.0
        
        # Config
        self.default_experiment_config_ = ExperimentConfig.FULL_RANGE_COVER.value
        self.declare_parameter('experiment_config', self.default_experiment_config_)
        self.experiment_config_int_ = self.get_parameter('experiment_config').value
        self.delay_config_ = ExperimentConfig(self.experiment_config_int_)
        
        # Load LSTM Model
        self.lstm_model_path_ = LSTM_MODEL_PATH
        # [NOTE] Input dim must match your training (15D: 7q + 7qd + 1delay)
        self.state_estimator_ = StateEstimator(input_dim_total=N_JOINTS*2+1).to(self.device_)
        self._load_model()

        # Delay Simulator
        self.delay_simulator_ = DelaySimulator(
            control_freq=self.control_freq_,
            config=self.delay_config_,
            seed= 100
        )
        
        # Buffers
        self.leader_q_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history_ = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)
        
        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        self.rnn_seq_len_ = RNN_SEQUENCE_LENGTH
        
        # Prefill
        self._prefill_buffers()
        self.is_leader_ready_ = False
        
        # Subscriber (Only need Local Robot)
        self.local_robot_state_subscriber_ = self.create_subscription(
            JointState, 'local_robot/joint_states', self.local_robot_state_callback, 10
        )
        
        # Timer
        self.control_timer_ = self.create_timer(self.dt_, self.test_loop_callback)
        self.get_logger().info("LSTM Test Node initialized.")

    def _load_model(self):
        try:
            lstm_ckpt = torch.load(self.lstm_model_path_, map_location=self.device_)
            # Handle both full checkpoint dict and state_dict only
            state_dict = lstm_ckpt.get('state_estimator_state_dict', lstm_ckpt)
            self.state_estimator_.load_state_dict(state_dict)
            self.state_estimator_.eval()
            self.get_logger().info(f"LSTM loaded from {self.lstm_model_path_}")
        except Exception as e:
            self.get_logger().fatal(f"Model load failed: {e}")
            raise

    def _prefill_buffers(self) -> None:
        q_init = INITIAL_JOINT_CONFIG.astype(np.float32)
        qd_init = np.zeros(N_JOINTS, dtype=np.float32)
        for _ in range(self.rnn_seq_len_ + 50):
            self.leader_q_history_.append(q_init)
            self.leader_qd_history_.append(qd_init)

    def local_robot_state_callback(self, msg: JointState) -> None:
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
        except (KeyError, IndexError):
            pass

    # ---------------------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------------------

    def _get_delayed_leader_data(self) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Returns:
            buffer: Flattened sequence for LSTM input (normalized delay)
            raw_delay_scalar: Raw delay steps (for logging)
            last_observation: The specific [q, qd] at t-delay (for reconstruction)
        """
        history_len = len(self.leader_q_history_)
        
        # 1. Get Raw Delay
        raw_delay_steps = int(self.delay_simulator_.get_observation_delay_steps(history_len))
        
        # 2. Normalize Delay (Divide by 100.0 to match training)
        normalized_delay = float(raw_delay_steps) / 100.0
        
        # 3. Calculate Indices
        most_recent_delayed_idx = -(raw_delay_steps + 1)
        oldest_idx = most_recent_delayed_idx - self.rnn_seq_len_ + 1
        
        buffer_seq = []
        
        # 4. Get Last Observation (for Residual Reconstruction)
        safe_last_idx = np.clip(most_recent_delayed_idx, -history_len, -1)
        last_q = self.leader_q_history_[safe_last_idx]
        last_qd = self.leader_qd_history_[safe_last_idx]
        last_observation = np.concatenate([last_q, last_qd])

        # 5. Build Sequence
        for i in range(oldest_idx, most_recent_delayed_idx + 1):
            safe_idx = np.clip(i, -history_len, -1)
            step_vector = np.concatenate([
                self.leader_q_history_[safe_idx],
                self.leader_qd_history_[safe_idx],
                [normalized_delay] # Use NORMALIZED delay
            ])
            buffer_seq.append(step_vector)
        
        buffer = np.array(buffer_seq).flatten().astype(np.float32)
        return buffer, float(raw_delay_steps), last_observation

    def _get_ground_truth(self) -> np.ndarray:
        """Current state of the local robot (Live)"""
        gt_q = self.leader_q_history_[-1]
        gt_qd = self.leader_qd_history_[-1]
        return np.concatenate([gt_q, gt_qd])

    # ---------------------------------------------------------------------
    # Test Loop
    # ---------------------------------------------------------------------

    def test_loop_callback(self) -> None:
        if not self.is_leader_ready_:
            return
            
        try:
            # 1. Get Inputs
            raw_delayed_sequence, raw_delay_scalar, last_observation = self._get_delayed_leader_data()
            
            # 2. Inference
            full_seq_t = torch.tensor(raw_delayed_sequence, dtype=torch.float32).to(self.device_).reshape(1, self.rnn_seq_len_, -1)

            with torch.no_grad():
                # Model output is the SCALED residual
                predicted_delta_scaled_t, _ = self.state_estimator_(full_seq_t)
            
            predicted_delta_scaled_np = predicted_delta_scaled_t.cpu().numpy().flatten()
            
            # 3. [CRITICAL FIX] Descaling: Delta = Scaled_Output / Scale_Factor
            predicted_delta_np = predicted_delta_scaled_np / TARGET_DELTA_SCALE
            
            # 4. Reconstruction: Target = Last_Obs + Delta
            predicted_np = last_observation + predicted_delta_np

            # 5. Ground Truth
            ground_truth_np = self._get_ground_truth()

            # 6. Metrics
            error_vec = ground_truth_np - predicted_np
            error_norm = np.linalg.norm(error_vec)
            
            # Logging
            print("\n" + "="*50)
            print(f"[LSTM Residual Test] Delay: {int(raw_delay_scalar)} steps")
            print("-" * 50)
            print(f"1. Ground Truth (t): \n   {np.round(ground_truth_np[:7], 4)}")
            print(f"2. Predicted (t):    \n   {np.round(predicted_np[:7], 4)}")
            print(f"   (Last Obs + Delta (Scaled/{TARGET_DELTA_SCALE}))")
            print("-" * 50)
            print(f"Prediction Error (L2 Norm): {error_norm:.5f}")
            print("="*50)

        except Exception as e:
            self.get_logger().error(f"Error in test loop: {e}")
            # import traceback
            # traceback.print_exc()

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