#!/usr/bin/env python3
"""
LSTM Accuracy Testing Script

Tests the trained LSTM model accuracy on different trajectory types to identify
if LSTM prediction error is causing the RL training plateau at 30mm.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime

class PositionLSTM(nn.Module):
    """Same LSTM architecture as in training environment"""
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class TrajectoryGenerator:
    """Generate different types of trajectories for testing"""
    
    @staticmethod
    def figure8_trajectory(t: np.ndarray, center: np.ndarray = None, scale: np.ndarray = None, freq: float = 0.1) -> np.ndarray:
        """Generate figure-8 trajectory (same as your training data)"""
        if center is None:
            center = np.array([0.4, 0.0, 0.6])
        if scale is None:
            scale = np.array([0.1, 0.3, 0.0])
        
        t_scaled = t * freq * 2 * np.pi
        dx = scale[0] * np.sin(t_scaled)
        dy = scale[1] * np.sin(t_scaled / 2)
        dz = scale[2] * np.sin(t_scaled / 3) if scale[2] > 0 else np.zeros_like(t)
        
        positions = np.column_stack([
            center[0] + dx,
            center[1] + dy, 
            center[2] + dz
        ])
        return positions
    
    @staticmethod
    def circular_trajectory(t: np.ndarray, center: np.ndarray = None, radius: float = 0.2, freq: float = 0.1) -> np.ndarray:
        """Generate circular trajectory"""
        if center is None:
            center = np.array([0.4, 0.0, 0.6])
        
        t_scaled = t * freq * 2 * np.pi
        dx = radius * np.cos(t_scaled)
        dy = radius * np.sin(t_scaled)
        dz = np.zeros_like(t)
        
        positions = np.column_stack([
            center[0] + dx,
            center[1] + dy,
            center[2] + dz
        ])
        return positions
    
    @staticmethod
    def random_walk_trajectory(t: np.ndarray, center: np.ndarray = None, step_size: float = 0.01, bounds: float = 0.3) -> np.ndarray:
        """Generate random walk trajectory"""
        if center is None:
            center = np.array([0.4, 0.0, 0.6])
        
        n_steps = len(t)
        positions = np.zeros((n_steps, 3))
        positions[0] = center
        
        for i in range(1, n_steps):
            # Random step
            step = np.random.normal(0, step_size, 3)
            new_pos = positions[i-1] + step
            
            # Keep within bounds
            new_pos = np.clip(new_pos, center - bounds, center + bounds)
            positions[i] = new_pos
        
        return positions
    
    @staticmethod
    def step_input_trajectory(t: np.ndarray, center: np.ndarray = None, step_magnitude: float = 0.2) -> np.ndarray:
        """Generate step input trajectory (sudden changes)"""
        if center is None:
            center = np.array([0.4, 0.0, 0.6])
        
        n_steps = len(t)
        positions = np.zeros((n_steps, 3))
        
        # Create step changes every 50 time steps
        step_interval = 50
        current_target = center.copy()
        
        for i in range(n_steps):
            if i % step_interval == 0 and i > 0:
                # Random step change
                step = np.random.uniform(-step_magnitude, step_magnitude, 3)
                current_target = center + step
            
            positions[i] = current_target
        
        return positions
    
    @staticmethod
    def linear_trajectory(t: np.ndarray, start: np.ndarray = None, end: np.ndarray = None) -> np.ndarray:
        """Generate linear trajectory"""
        if start is None:
            start = np.array([0.3, -0.2, 0.5])
        if end is None:
            end = np.array([0.5, 0.2, 0.7])
        
        n_steps = len(t)
        positions = np.zeros((n_steps, 3))
        
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            positions[i] = start + alpha * (end - start)
        
        return positions

class LSTMAccuracyTester:
    """Test LSTM accuracy on various trajectory types"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.model = self._load_model(model_path)
        self.history_length = 10  # Same as in RL environment
        
    def _load_model(self, model_path: str) -> PositionLSTM:
        """Load the trained LSTM model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = PositionLSTM(input_size=3, hidden_size=128, num_layers=2).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded LSTM model from: {model_path}")
        return model
    
    def test_trajectory(self, positions: np.ndarray, trajectory_name: str) -> Dict:
        """Test LSTM accuracy on a given trajectory"""
        n_points = len(positions)
        if n_points < self.history_length + 10:
            raise ValueError(f"Trajectory too short. Need at least {self.history_length + 10} points")
        
        prediction_errors = []
        multi_step_errors = {1: [], 2: [], 3: [], 5: []}
        
        # Test single-step prediction accuracy
        for i in range(self.history_length, n_points - 5):
            # Get history window
            history = positions[i - self.history_length:i]
            actual_next = positions[i]
            
            # Predict next position
            input_tensor = torch.from_numpy(history.astype(np.float32)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predicted_next = self.model(input_tensor).squeeze(0).cpu().numpy()
            
            # Calculate error in mm
            error_mm = np.linalg.norm(predicted_next - actual_next) * 1000
            prediction_errors.append(error_mm)
            
            # Multi-step prediction errors
            for steps_ahead in multi_step_errors.keys():
                if i + steps_ahead < n_points:
                    actual_future = positions[i + steps_ahead]
                    
                    # For multi-step, we need to predict iteratively
                    current_history = history.copy()
                    for step in range(steps_ahead):
                        input_tensor = torch.from_numpy(current_history[-self.history_length:].astype(np.float32)).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            next_pred = self.model(input_tensor).squeeze(0).cpu().numpy()
                        # Update history with prediction
                        current_history = np.vstack([current_history, next_pred])
                    
                    final_prediction = current_history[-1]
                    error_mm = np.linalg.norm(final_prediction - actual_future) * 1000
                    multi_step_errors[steps_ahead].append(error_mm)
        
        # Calculate statistics
        results = {
            'trajectory_name': trajectory_name,
            'n_predictions': len(prediction_errors),
            'single_step': {
                'mean_error_mm': np.mean(prediction_errors),
                'std_error_mm': np.std(prediction_errors),
                'max_error_mm': np.max(prediction_errors),
                'min_error_mm': np.min(prediction_errors),
                'median_error_mm': np.median(prediction_errors),
                'errors': prediction_errors
            },
            'multi_step': {}
        }
        
        for steps, errors in multi_step_errors.items():
            if errors:
                results['multi_step'][f'{steps}_step'] = {
                    'mean_error_mm': np.mean(errors),
                    'std_error_mm': np.std(errors),
                    'max_error_mm': np.max(errors),
                    'errors': errors
                }
        
        return results
    
    def run_comprehensive_test(self, duration: float = 60.0, freq: float = 200.0) -> Dict:
        """Run comprehensive accuracy test on multiple trajectory types"""
        print(f"Running comprehensive LSTM accuracy test...")
        print(f"Duration: {duration}s, Frequency: {freq}Hz")
        print("="*60)
        
        # Generate time vector
        t = np.linspace(0, duration, int(duration * freq))
        
        # Test different trajectory types
        trajectories = {
            'figure8_original': TrajectoryGenerator.figure8_trajectory(t),
            'figure8_fast': TrajectoryGenerator.figure8_trajectory(t, freq=0.2),
            'figure8_slow': TrajectoryGenerator.figure8_trajectory(t, freq=0.05),
            'circular': TrajectoryGenerator.circular_trajectory(t),
            'random_walk': TrajectoryGenerator.random_walk_trajectory(t),
            'step_input': TrajectoryGenerator.step_input_trajectory(t),
            'linear': TrajectoryGenerator.linear_trajectory(t)
        }
        
        results = {}
        
        for traj_name, positions in trajectories.items():
            print(f"Testing trajectory: {traj_name}")
            try:
                result = self.test_trajectory(positions, traj_name)
                results[traj_name] = result
                
                # Print summary
                single_step = result['single_step']
                print(f"  Single-step error: {single_step['mean_error_mm']:.1f}±{single_step['std_error_mm']:.1f}mm")
                print(f"  Max error: {single_step['max_error_mm']:.1f}mm")
                
                if 'multi_step' in result and '5_step' in result['multi_step']:
                    five_step = result['multi_step']['5_step']
                    print(f"  5-step ahead error: {five_step['mean_error_mm']:.1f}±{five_step['std_error_mm']:.1f}mm")
                
            except Exception as e:
                print(f"  Error testing {traj_name}: {e}")
            
            print()
        
        return results
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot comprehensive results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data for plotting
        traj_names = list(results.keys())
        single_step_means = [results[name]['single_step']['mean_error_mm'] for name in traj_names]
        single_step_stds = [results[name]['single_step']['std_error_mm'] for name in traj_names]
        
        # 1. Single-step prediction errors by trajectory
        axes[0, 0].bar(range(len(traj_names)), single_step_means, yerr=single_step_stds, capsize=5)
        axes[0, 0].set_xticks(range(len(traj_names)))
        axes[0, 0].set_xticklabels(traj_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Prediction Error (mm)')
        axes[0, 0].set_title('Single-Step Prediction Error by Trajectory')
        axes[0, 0].axhline(y=30, color='red', linestyle='--', label='RL Plateau (30mm)')
        axes[0, 0].legend()
        
        # 2. Multi-step prediction errors for figure8_original
        if 'figure8_original' in results and 'multi_step' in results['figure8_original']:
            multi_step_data = results['figure8_original']['multi_step']
            steps = [int(k.split('_')[0]) for k in multi_step_data.keys()]
            multi_errors = [multi_step_data[f'{s}_step']['mean_error_mm'] for s in steps]
            
            axes[0, 1].plot(steps, multi_errors, 'o-', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Steps Ahead')
            axes[0, 1].set_ylabel('Prediction Error (mm)')
            axes[0, 1].set_title('Multi-Step Prediction Error (Figure-8)')
            axes[0, 1].axhline(y=30, color='red', linestyle='--', label='RL Plateau (30mm)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution for figure8_original
        if 'figure8_original' in results:
            errors = results['figure8_original']['single_step']['errors']
            axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=30, color='red', linestyle='--', linewidth=2, label='RL Plateau (30mm)')
            axes[1, 0].axvline(x=np.mean(errors), color='blue', linestyle='-', linewidth=2, label=f'Mean ({np.mean(errors):.1f}mm)')
            axes[1, 0].set_xlabel('Prediction Error (mm)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Error Distribution (Figure-8 Original)')
            axes[1, 0].legend()
        
        # 4. Comparison with different trajectory frequencies
        freq_results = {name: results[name] for name in results.keys() if 'figure8' in name}
        if len(freq_results) > 1:
            freq_names = list(freq_results.keys())
            freq_errors = [freq_results[name]['single_step']['mean_error_mm'] for name in freq_names]
            
            axes[1, 1].bar(range(len(freq_names)), freq_errors)
            axes[1, 1].set_xticks(range(len(freq_names)))
            axes[1, 1].set_xticklabels(freq_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Prediction Error (mm)')
            axes[1, 1].set_title('Figure-8 Frequency Sensitivity')
            axes[1, 1].axhline(y=30, color='red', linestyle='--', label='RL Plateau (30mm)')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, save_path: str):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for traj_name, result in results.items():
            results_serializable[traj_name] = {
                'trajectory_name': result['trajectory_name'],
                'n_predictions': result['n_predictions'],
                'single_step': {
                    k: v if not isinstance(v, list) else v 
                    for k, v in result['single_step'].items()
                }
            }
            if 'multi_step' in result:
                results_serializable[traj_name]['multi_step'] = result['multi_step']
        
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"Results saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Test LSTM accuracy on various trajectories")
    parser.add_argument("--model_path", type=str, 
                       default="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays/state_predictor/models/config1/LSTM_Model_Config1_best.pth",
                       help="Path to trained LSTM model")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    parser.add_argument("--freq", type=float, default=200.0, help="Sampling frequency in Hz")
    parser.add_argument("--output_dir", type=str, default="./lstm_accuracy_results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = LSTMAccuracyTester(args.model_path, args.device)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(args.duration, args.freq)
    
    # Generate output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(args.output_dir, f"lstm_accuracy_results_{timestamp}.png")
    json_path = os.path.join(args.output_dir, f"lstm_accuracy_results_{timestamp}.json")
    
    # Save and plot results
    tester.save_results(results, json_path)
    tester.plot_results(results, plot_path)
    
    # Print summary
    print("\n" + "="*60)
    print("LSTM ACCURACY TEST SUMMARY")
    print("="*60)
    
    figure8_result = results.get('figure8_original', {})
    if figure8_result:
        single_step = figure8_result['single_step']
        print(f"Figure-8 (Original Training Data):")
        print(f"  Mean prediction error: {single_step['mean_error_mm']:.1f} ± {single_step['std_error_mm']:.1f}mm")
        print(f"  Max prediction error: {single_step['max_error_mm']:.1f}mm")
        
        if single_step['mean_error_mm'] > 25:
            print("\n⚠️  WARNING: LSTM prediction error > 25mm")
            print("   This could explain the 30mm RL plateau!")
            print("   Consider retraining LSTM with more diverse data.")
        elif single_step['mean_error_mm'] < 10:
            print("\n✅ LSTM prediction accuracy is good")
            print("   Plateau likely caused by other factors (exploration, reward function, etc.)")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()