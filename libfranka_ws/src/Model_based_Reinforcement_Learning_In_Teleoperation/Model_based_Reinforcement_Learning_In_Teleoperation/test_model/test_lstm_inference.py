"""
LSTM State Estimator Testing with Real ROS2 Leader Robot.

This script subscribes to the actual leader robot joint states published by
local_robot_sim.py and tests LSTM inference on real robot data.

Pipeline:
1. Launch ROS2 leader robot node (local_robot_sim.py)
2. Subscribe to /local_robot/joint_states topic
3. Collect actual joint commands and build delayed buffers
4. Run LSTM inference
5. Compare predictions with ground truth
6. Generate performance metrics and visualizations
"""

import os
import sys
import torch
import numpy as np
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Deque
from collections import deque
import json

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    LSTM_MODEL_PATH,
    N_JOINTS,
    RNN_SEQUENCE_LENGTH,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    DEFAULT_CONTROL_FREQ,
)


class LSTMTestingNode(Node):
    """ROS2 Node for testing LSTM on real leader robot data."""
    
    def __init__(self, model_path: str, output_dir: str, num_samples: int = 3000):
        super().__init__('lstm_testing_node')
        
        self.model_path = model_path
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.sample_count = 0
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load LSTM model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_estimator = self._load_model()
        
        # Data collection
        self.joint_buffer: Deque = deque(maxlen=RNN_SEQUENCE_LENGTH * 2 * 100)
        self.episode_data = {
            'time': [],
            'true_q': [],
            'true_qd': [],
            'predicted_q': [],
            'predicted_qd': [],
            'joint_error': [],
        }
        
        self.last_time = None
        self.start_time = None
        
        # Subscribe to leader robot joint states
        self.subscription = self.create_subscription(
            JointState,
            'local_robot/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.logger.info(f"LSTM Testing Node initialized. Model: {model_path}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Collecting {num_samples} samples")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        log_file = os.path.join(self.output_dir, "lstm_ros_test.log")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        return logging.getLogger(__name__)
    
    def _load_model(self) -> StateEstimator:
        """Load pre-trained LSTM model."""
        self.logger.info(f"Loading LSTM model from: {self.model_path}")
        
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
        self.logger.info("Model loaded successfully")
        return state_estimator
    
    def joint_state_callback(self, msg: JointState) -> None:
        """Callback for joint state messages."""
        
        if self.start_time is None:
            self.start_time = self.get_clock().now()
        
        try:
            # Extract joint data
            q_current = np.array(msg.position, dtype=np.float32)
            qd_current = np.array(msg.velocity, dtype=np.float32)
            
            # Stack into [q, qd] format
            joint_state = np.concatenate([q_current, qd_current])
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
            current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.episode_data['time'].append(current_time)
            self.episode_data['true_q'].append(true_target[:N_JOINTS].copy())
            self.episode_data['true_qd'].append(true_target[N_JOINTS:].copy())
            self.episode_data['predicted_q'].append(predicted_target[:N_JOINTS].copy())
            self.episode_data['predicted_qd'].append(predicted_target[N_JOINTS:].copy())
            self.episode_data['joint_error'].append(q_error)
            
            self.sample_count += 1
            
            if self.sample_count % 500 == 0:
                self.logger.info(f"Collected {self.sample_count}/{self.num_samples} samples")
            
            # Stop when we have enough data
            if self.sample_count >= self.num_samples:
                self.logger.info("Target number of samples reached. Processing results...")
                self.process_and_save_results()
                rclpy.shutdown()
        
        except Exception as e:
            self.logger.error(f"Error in callback: {e}")
            import traceback
            traceback.print_exc()
    
    def process_and_save_results(self) -> None:
        """Process collected data and save results."""
        
        self.logger.info("=" * 80)
        self.logger.info("PROCESSING RESULTS")
        self.logger.info("=" * 80)
        
        # Convert to numpy arrays
        for key in self.episode_data:
            if isinstance(self.episode_data[key], list) and len(self.episode_data[key]) > 0:
                if isinstance(self.episode_data[key][0], np.ndarray):
                    self.episode_data[key] = np.array(self.episode_data[key])
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Save results
        self._save_metrics(metrics)
        self._plot_results(metrics)
        
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"Position RMSE: {metrics['joint_position_rmse']:.8f} rad")
        self.logger.info(f"Velocity RMSE: {metrics['joint_velocity_rmse']:.8f} rad/s")
    
    def _compute_metrics(self) -> Dict:
        """Compute accuracy metrics."""
        metrics = {}
        
        true_q = np.array(self.episode_data['true_q'])
        predicted_q = np.array(self.episode_data['predicted_q'])
        true_qd = np.array(self.episode_data['true_qd'])
        predicted_qd = np.array(self.episode_data['predicted_qd'])
        
        q_error_l2 = np.linalg.norm(true_q - predicted_q, axis=1)
        qd_error_l2 = np.linalg.norm(true_qd - predicted_qd, axis=1)
        
        metrics['joint_position_mse'] = np.mean((true_q - predicted_q) ** 2)
        metrics['joint_position_rmse'] = np.sqrt(metrics['joint_position_mse'])
        metrics['joint_position_mae'] = np.mean(np.abs(true_q - predicted_q))
        metrics['joint_position_max_error'] = np.max(q_error_l2)
        metrics['joint_position_mean_l2_error'] = np.mean(q_error_l2)
        
        metrics['joint_velocity_mse'] = np.mean((true_qd - predicted_qd) ** 2)
        metrics['joint_velocity_rmse'] = np.sqrt(metrics['joint_velocity_mse'])
        metrics['joint_velocity_mae'] = np.mean(np.abs(true_qd - predicted_qd))
        metrics['joint_velocity_max_error'] = np.max(qd_error_l2)
        metrics['joint_velocity_mean_l2_error'] = np.mean(qd_error_l2)
        
        for j in range(N_JOINTS):
            metrics[f'joint_{j}_position_rmse'] = np.sqrt(
                np.mean((true_q[:, j] - predicted_q[:, j]) ** 2)
            )
            metrics[f'joint_{j}_velocity_rmse'] = np.sqrt(
                np.mean((true_qd[:, j] - predicted_qd[:, j]) ** 2)
            )
        
        return metrics
    
    def _save_metrics(self, metrics: Dict) -> None:
        """Save metrics to JSON and text files."""
        json_path = os.path.join(self.output_dir, 'metrics.json')
        metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in metrics.items()}
        with open(json_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        txt_path = os.path.join(self.output_dir, 'metrics_report.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LSTM STATE ESTIMATOR TEST - REAL ROS2 DATA\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples collected: {self.sample_count}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("JOINT POSITION ACCURACY\n")
            f.write("-" * 80 + "\n")
            f.write(f"MSE:  {metrics['joint_position_mse']:.8f} rad²\n")
            f.write(f"RMSE: {metrics['joint_position_rmse']:.8f} rad\n")
            f.write(f"MAE:  {metrics['joint_position_mae']:.8f} rad\n")
            f.write(f"Max Error: {metrics['joint_position_max_error']:.8f} rad\n")
            f.write(f"Mean L2 Error: {metrics['joint_position_mean_l2_error']:.8f} rad\n\n")
            
            f.write("JOINT VELOCITY ACCURACY\n")
            f.write("-" * 80 + "\n")
            f.write(f"MSE:  {metrics['joint_velocity_mse']:.8f} (rad/s)²\n")
            f.write(f"RMSE: {metrics['joint_velocity_rmse']:.8f} rad/s\n")
            f.write(f"MAE:  {metrics['joint_velocity_mae']:.8f} rad/s\n")
            f.write(f"Max Error: {metrics['joint_velocity_max_error']:.8f} rad/s\n")
            f.write(f"Mean L2 Error: {metrics['joint_velocity_mean_l2_error']:.8f} rad/s\n\n")
            
            f.write("PER-JOINT POSITION RMSE\n")
            f.write("-" * 80 + "\n")
            for j in range(N_JOINTS):
                rmse = metrics[f'joint_{j}_position_rmse']
                f.write(f"Joint {j}: {rmse:.8f} rad\n")
            f.write("\n")
            
            f.write("PER-JOINT VELOCITY RMSE\n")
            f.write("-" * 80 + "\n")
            for j in range(N_JOINTS):
                rmse = metrics[f'joint_{j}_velocity_rmse']
                f.write(f"Joint {j}: {rmse:.8f} rad/s\n")
            f.write("\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Metrics saved to {json_path} and {txt_path}")
    
    def _plot_results(self, metrics: Dict) -> None:
        """Generate visualization plots."""
        time = np.array(self.episode_data['time'])
        true_q = np.array(self.episode_data['true_q'])
        predicted_q = np.array(self.episode_data['predicted_q'])
        true_qd = np.array(self.episode_data['true_qd'])
        predicted_qd = np.array(self.episode_data['predicted_qd'])
        
        # Plot joint positions
        fig, axes = plt.subplots(7, 1, figsize=(14, 12))
        for j in range(N_JOINTS):
            ax = axes[j]
            ax.plot(time, true_q[:, j], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
            ax.plot(time, predicted_q[:, j], 'r--', linewidth=1.5, label='LSTM Prediction', alpha=0.8)
            ax.fill_between(time, true_q[:, j], predicted_q[:, j], alpha=0.2, color='gray')
            
            ax.set_ylabel(f'Joint {j} (rad)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
            rmse = metrics[f'joint_{j}_position_rmse']
            ax.set_title(f'Joint {j} Position Tracking (RMSE: {rmse:.6f} rad)', fontsize=10)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'joint_positions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot velocities
        fig, axes = plt.subplots(7, 1, figsize=(14, 12))
        for j in range(N_JOINTS):
            ax = axes[j]
            ax.plot(time, true_qd[:, j], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
            ax.plot(time, predicted_qd[:, j], 'r--', linewidth=1.5, label='LSTM Prediction', alpha=0.8)
            ax.fill_between(time, true_qd[:, j], predicted_qd[:, j], alpha=0.2, color='gray')
            
            ax.set_ylabel(f'Joint {j} (rad/s)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
            rmse = metrics[f'joint_{j}_velocity_rmse']
            ax.set_title(f'Joint {j} Velocity Tracking (RMSE: {rmse:.6f} rad/s)', fontsize=10)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'joint_velocities.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Plots saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Test LSTM on real ROS2 leader robot data")
    parser.add_argument("--model-path", type=str, default=LSTM_MODEL_PATH, help="Path to LSTM model")
    parser.add_argument("--output-dir", type=str, default="./lstm_ros_test_output", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=3000, help="Number of samples to collect")
    
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    rclpy.init()
    
    try:
        testing_node = LSTMTestingNode(
            model_path=args.model_path,
            output_dir=output_dir,
            num_samples=args.num_samples
        )
        
        testing_node.get_logger().info("\n" + "=" * 80)
        testing_node.get_logger().info("LSTM TESTING WITH REAL ROS2 DATA")
        testing_node.get_logger().info("=" * 80)
        testing_node.get_logger().info("Waiting for joint state messages...")
        testing_node.get_logger().info("Make sure local_robot_sim.py is running!")
        testing_node.get_logger().info("=" * 80 + "\n")
        
        rclpy.spin(testing_node)
    
    except KeyboardInterrupt:
        print("\n[INFO] Testing interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()