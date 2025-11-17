#!/usr/bin/env python3
"""
LSTM Testing with REAL ROS2 Leader Robot Data

This script collects real joint states from local_robot_sim.py and tests
the LSTM on that actual data, showing true performance.

Usage:
    Terminal 1: python3 local_robot_sim.py
    Terminal 2: python3 test_lstm_with_ros2_data.py \
        --model-path /path/to/model.pth \
        --num-samples 3000 \
        --output results.csv
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import torch
import numpy as np
import argparse
import os
import json
import csv
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    LSTM_MODEL_PATH,
    N_JOINTS,
    RNN_SEQUENCE_LENGTH,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
)


class LSTMTesterROS2(Node):
    """Test LSTM using real ROS2 joint state data."""
    
    def __init__(self, model_path, num_samples, output_dir):
        super().__init__('lstm_tester_ros2')
        
        self.model_path = model_path
        self.num_samples_target = num_samples
        self.output_dir = output_dir
        self.sample_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load LSTM model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_estimator = self._load_model()
        
        # Data collection
        self.joint_buffer = deque(maxlen=RNN_SEQUENCE_LENGTH)
        self.episode_data = {
            'time': [],
            'true_q': [],
            'true_qd': [],
            'predicted_q': [],
            'predicted_qd': [],
            'joint_error': [],
        }
        
        # CSV writer
        self.csv_file = open(os.path.join(output_dir, 'ros2_lstm_test.csv'), 'w', newline='')
        header = ['Step', 'Time'] + [f'TrueQ{i}' for i in range(7)] + [f'PredQ{i}' for i in range(7)] + \
                 [f'TrueQD{i}' for i in range(7)] + [f'PredQD{i}' for i in range(7)] + ['Error']
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=header)
        self.csv_writer.writeheader()
        
        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState,
            'local_robot/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.get_logger().info("="*150)
        self.get_logger().info("LSTM TESTING WITH REAL ROS2 DATA")
        self.get_logger().info("="*150)
        self.get_logger().info(f"Model: {model_path}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Target samples: {num_samples}")
        self.get_logger().info("\nCollecting data...\n")
        
        # Print header
        print(f"{'Step':<8} {'Time':<12} {'TrueQ0':<10} {'PredQ0':<10} {'ErrQ0':<10} {'TrueQD0':<10} {'PredQD0':<10} {'ErrQD0':<10}")
        print("-"*100)
    
    def _load_model(self) -> StateEstimator:
        """Load pre-trained LSTM model."""
        self.get_logger().info(f"Loading LSTM model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        
        state_estimator = StateEstimator(
            input_dim=N_JOINTS * 2,
            hidden_dim=RNN_HIDDEN_DIM,
            num_layers=RNN_NUM_LAYERS,
            output_dim=N_JOINTS * 2,
        ).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'state_estimator_state_dict' in checkpoint:
            state_estimator.load_state_dict(checkpoint['state_estimator_state_dict'])
        else:
            state_estimator.load_state_dict(checkpoint)
        
        state_estimator.eval()
        self.get_logger().info("Model loaded successfully\n")
        return state_estimator
    
    def joint_state_callback(self, msg: JointState):
        """Process incoming joint states."""
        
        try:
            # Extract data
            q = np.array(msg.position, dtype=np.float32)
            qd = np.array(msg.velocity, dtype=np.float32)
            
            # Get timestamp
            timestamp = msg.header.stamp
            time_sec = timestamp.sec + timestamp.nanosec / 1e9
            
            # Create joint state vector [q, qd]
            joint_state = np.concatenate([q, qd])
            self.joint_buffer.append(joint_state)
            
            # Need enough history for LSTM sequence
            if len(self.joint_buffer) < RNN_SEQUENCE_LENGTH:
                return
            
            # Build delayed sequence
            delayed_seq = np.array(list(self.joint_buffer)[-RNN_SEQUENCE_LENGTH:])
            
            # Run LSTM inference
            with torch.no_grad():
                delayed_seq_tensor = torch.tensor(
                    delayed_seq.reshape(1, RNN_SEQUENCE_LENGTH, N_JOINTS * 2),
                    dtype=torch.float32,
                    device=self.device
                )
                predicted_target, _ = self.state_estimator(delayed_seq_tensor)
                predicted_target = predicted_target.cpu().numpy()[0]
            
            # Compute error
            true_target = joint_state
            q_error = np.linalg.norm(true_target[:N_JOINTS] - predicted_target[:N_JOINTS])
            
            # Store data
            self.episode_data['time'].append(time_sec)
            self.episode_data['true_q'].append(true_target[:N_JOINTS].copy())
            self.episode_data['true_qd'].append(true_target[N_JOINTS:].copy())
            self.episode_data['predicted_q'].append(predicted_target[:N_JOINTS].copy())
            self.episode_data['predicted_qd'].append(predicted_target[N_JOINTS:].copy())
            self.episode_data['joint_error'].append(q_error)
            
            # Write to CSV
            row_data = {
                'Step': self.sample_count,
                'Time': f"{time_sec:.6f}",
                'Error': f"{q_error:.6f}"
            }
            for i in range(7):
                row_data[f'TrueQ{i}'] = f"{true_target[i]:.6f}"
                row_data[f'PredQ{i}'] = f"{predicted_target[i]:.6f}"
                row_data[f'TrueQD{i}'] = f"{true_target[N_JOINTS+i]:.6f}"
                row_data[f'PredQD{i}'] = f"{predicted_target[N_JOINTS+i]:.6f}"
            self.csv_writer.writerow(row_data)
            self.csv_file.flush()
            
            # Print real-time data (every sample)
            true_q0 = true_target[0]
            pred_q0 = predicted_target[0]
            err_q0 = abs(true_q0 - pred_q0)
            true_qd0 = true_target[N_JOINTS]
            pred_qd0 = predicted_target[N_JOINTS]
            err_qd0 = abs(true_qd0 - pred_qd0)
            
            print(f"{self.sample_count:<8} {time_sec:<12.3f} {true_q0:<10.6f} {pred_q0:<10.6f} {err_q0:<10.6f} {true_qd0:<10.6f} {pred_qd0:<10.6f} {err_qd0:<10.6f}")
            
            self.sample_count += 1
            
            # Check if we've collected enough samples
            if self.sample_count % 500 == 0:
                self.get_logger().info(f"Collected {self.sample_count}/{self.num_samples_target} samples")
            
            if self.sample_count >= self.num_samples_target:
                self.get_logger().info(f"\nCollected {self.sample_count} samples. Processing results...")
                self._process_results()
                rclpy.shutdown()
        
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_results(self):
        """Process and save results."""
        self.csv_file.close()
        
        # Convert to numpy arrays
        times = np.array(self.episode_data['time'])
        true_q = np.array(self.episode_data['true_q'])
        predicted_q = np.array(self.episode_data['predicted_q'])
        true_qd = np.array(self.episode_data['true_qd'])
        predicted_qd = np.array(self.episode_data['predicted_qd'])
        
        # Compute metrics
        metrics = {
            'position_rmse': np.sqrt(np.mean((true_q - predicted_q) ** 2)),
            'position_mae': np.mean(np.abs(true_q - predicted_q)),
            'velocity_rmse': np.sqrt(np.mean((true_qd - predicted_qd) ** 2)),
            'velocity_mae': np.mean(np.abs(true_qd - predicted_qd)),
        }
        
        for j in range(7):
            metrics[f'joint_{j}_pos_rmse'] = np.sqrt(np.mean((true_q[:, j] - predicted_q[:, j]) ** 2))
            metrics[f'joint_{j}_vel_rmse'] = np.sqrt(np.mean((true_qd[:, j] - predicted_qd[:, j]) ** 2))
        
        # Save metrics
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print results
        print("\n" + "="*100)
        print("LSTM TEST RESULTS - REAL ROS2 DATA")
        print("="*100)
        print(f"Samples: {self.sample_count}")
        print(f"Duration: {times[-1] - times[0]:.3f} seconds")
        print(f"\nPosition RMSE: {metrics['position_rmse']:.8f} rad")
        print(f"Position MAE:  {metrics['position_mae']:.8f} rad")
        print(f"\nVelocity RMSE: {metrics['velocity_rmse']:.8f} rad/s")
        print(f"Velocity MAE:  {metrics['velocity_mae']:.8f} rad/s")
        
        print("\nPer-Joint Position RMSE:")
        for j in range(7):
            print(f"  Joint {j}: {metrics[f'joint_{j}_pos_rmse']:.8f} rad")
        
        print("\nResults saved to: " + self.output_dir)
        print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Test LSTM with real ROS2 data')
    parser.add_argument('--model-path', type=str, default=LSTM_MODEL_PATH, help='Path to LSTM model')
    parser.add_argument('--num-samples', type=int, default=3000, help='Number of samples to collect')
    parser.add_argument('--output-dir', type=str, default='./lstm_ros2_test_results', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    rclpy.init()
    
    try:
        tester = LSTMTesterROS2(
            model_path=args.model_path,
            num_samples=args.num_samples,
            output_dir=output_dir
        )
        
        rclpy.spin(tester)
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()