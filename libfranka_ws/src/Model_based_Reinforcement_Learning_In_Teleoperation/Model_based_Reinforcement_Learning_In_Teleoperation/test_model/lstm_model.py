"""
LSTM Test - UNIVERSAL FIX

This script tries multiple ways to initialize StateEstimator
to work with any implementation.
"""

import os
import torch
import numpy as np
import argparse
import sys
from typing import Dict
import time
import inspect

print("Importing modules...")

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, make_vec_env
except ImportError:
    print("Warning: stable_baselines3 not available")

try:
    from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
    from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
    from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
    from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator
    from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
        N_JOINTS,
        RNN_SEQUENCE_LENGTH,
        RNN_HIDDEN_DIM,
        RNN_NUM_LAYERS,
    )
    print("All modules imported successfully!")
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)


def inspect_state_estimator():
    """Print what StateEstimator actually expects."""
    print("\n" + "="*70)
    print("StateEstimator Inspection")
    print("="*70)
    
    sig = inspect.signature(StateEstimator.__init__)
    print(f"Signature: {sig}")
    
    params = list(sig.parameters.keys())
    params.remove('self')
    
    print(f"\nParameters (excluding self):")
    for i, param in enumerate(params, 1):
        print(f"  {i}. {param}")
    
    print(f"\nTotal parameters: {len(params)}")
    return params


def create_state_estimator(device):
    """Try multiple ways to create StateEstimator."""
    print(f"\nAttempting to create StateEstimator...")
    
    # First, inspect what we need
    params = inspect_state_estimator()
    
    attempts = []
    
    # Attempt 1: positional arguments in order
    if len(params) >= 4:
        attempts.append({
            'name': 'Positional (14, 512, 4, 14)',
            'args': [N_JOINTS * 2, RNN_HIDDEN_DIM, RNN_NUM_LAYERS, N_JOINTS * 2],
            'kwargs': {}
        })
    
    # Attempt 2: with possible parameter names
    attempts.append({
        'name': 'Keyword: input_dim, hidden_dim, num_layers, output_dim',
        'args': [],
        'kwargs': {
            'input_dim': N_JOINTS * 2,
            'hidden_dim': RNN_HIDDEN_DIM,
            'num_layers': RNN_NUM_LAYERS,
            'output_dim': N_JOINTS * 2
        }
    })
    
    attempts.append({
        'name': 'Keyword: state_size, hidden_size, num_layers, output_size',
        'args': [],
        'kwargs': {
            'state_size': N_JOINTS * 2,
            'hidden_size': RNN_HIDDEN_DIM,
            'num_layers': RNN_NUM_LAYERS,
            'output_size': N_JOINTS * 2
        }
    })
    
    attempts.append({
        'name': 'Keyword: input_size, hidden_size, num_layers, output_size',
        'args': [],
        'kwargs': {
            'input_size': N_JOINTS * 2,
            'hidden_size': RNN_HIDDEN_DIM,
            'num_layers': RNN_NUM_LAYERS,
            'output_size': N_JOINTS * 2
        }
    })
    
    for attempt in attempts:
        try:
            print(f"\n  Trying: {attempt['name']}")
            if attempt['args']:
                estimator = StateEstimator(*attempt['args']).to(device)
            else:
                estimator = StateEstimator(**attempt['kwargs']).to(device)
            print(f"  SUCCESS!")
            return estimator
        except TypeError as e:
            print(f"  Failed: {e}")
    
    raise RuntimeError(f"Could not create StateEstimator with any approach. Parameters: {params}")


class LSTMTester:
    """LSTM tester with universal StateEstimator initialization."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize tester."""
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"\nLoading model from: {self.model_path}")
        print(f"Using device: {self.device}")
        
        self.state_estimator = self._load_model()
        
    def _load_model(self) -> StateEstimator:
        """Load model with universal initialization."""
        
        self.state_estimator = create_state_estimator(self.device)
        
        print(f"\nLoading checkpoint...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_estimator_state_dict' in checkpoint:
                self.state_estimator.load_state_dict(checkpoint['state_estimator_state_dict'])
                print(f"Loaded: state_estimator_state_dict")
            elif 'model_state_dict' in checkpoint:
                self.state_estimator.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded: model_state_dict")
            elif 'state_dict' in checkpoint:
                self.state_estimator.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded: state_dict")
            else:
                # Try to load as direct state dict
                self.state_estimator.load_state_dict(checkpoint)
                print(f"Loaded: checkpoint directly as state_dict")
        else:
            self.state_estimator.load_state_dict(checkpoint)
            print(f"Loaded: checkpoint directly")
        
        self.state_estimator.eval()
        print("Model ready!")
        return self.state_estimator
    
    def _create_environment(self, delay_config, trajectory_type, seed=42):
        """Create test environment."""
        def make_env():
            return TeleoperationEnvWithDelay(
                delay_config=delay_config,
                trajectory_type=trajectory_type,
                randomize_trajectory=False,
                seed=seed
            )
        return make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=seed)
    
    def test(self, delay_config, trajectory_type, num_steps=100):
        """Run test."""
        print(f"\n{'='*70}")
        print(f"Testing: {trajectory_type.value} with {delay_config.name}")
        print(f"Steps: {num_steps}")
        print(f"{'='*70}\n")
        
        env = self._create_environment(delay_config, trajectory_type)
        obs = env.reset()
        
        position_errors = []
        velocity_errors = []
        inference_times = []
        
        with torch.no_grad():
            for step in range(num_steps):
                delayed_buffers = env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
                true_targets = env.env_method("get_true_current_target")
                
                delayed_seq = np.array([
                    buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2)
                    for buf in delayed_buffers
                ])[0]
                true_target = np.array(true_targets)[0]
                
                delayed_seq_tensor = torch.tensor(
                    delayed_seq.reshape(1, RNN_SEQUENCE_LENGTH, N_JOINTS * 2),
                    dtype=torch.float32,
                    device=self.device
                )
                
                start = time.perf_counter()
                predicted_target, _ = self.state_estimator(delayed_seq_tensor)
                inference_time = (time.perf_counter() - start) * 1000
                
                predicted_target = predicted_target.cpu().numpy()[0]
                
                pos_error = np.linalg.norm(
                    predicted_target[:N_JOINTS] - true_target[:N_JOINTS]
                )
                vel_error = np.linalg.norm(
                    predicted_target[N_JOINTS:] - true_target[N_JOINTS:]
                )
                
                position_errors.append(pos_error)
                velocity_errors.append(vel_error)
                inference_times.append(inference_time)
                
                random_action = np.array([env.action_space.sample()])
                obs, _, _, _ = env.step(random_action)
                
                if (step + 1) % max(1, num_steps // 5) == 0:
                    print(f"Step {step + 1}/{num_steps}: Pos={pos_error:.5f} Vel={vel_error:.5f}")
        
        env.close()
        
        position_errors = np.array(position_errors)
        velocity_errors = np.array(velocity_errors)
        inference_times = np.array(inference_times)
        
        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"{'='*70}")
        print(f"Position Error: {np.mean(position_errors):.6f} ± {np.std(position_errors):.6f} rad")
        print(f"               max: {np.max(position_errors):.6f}")
        print(f"\nVelocity Error: {np.mean(velocity_errors):.6f} ± {np.std(velocity_errors):.6f} rad/s")
        print(f"               max: {np.max(velocity_errors):.6f}")
        print(f"\nInference Time: {np.mean(inference_times):.3f} ± {np.std(inference_times):.3f} ms")
        print(f"               max: {np.max(inference_times):.3f} ms")
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LSTM State Estimator Test")
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--trajectory", type=str, default="figure_8", 
                       choices=['figure_8', 'circular', 'linear'])
    parser.add_argument("--delay-config", type=str, default="3", choices=['1', '2', '3', '4'])
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], 
                  '3': config_options[2], '4': config_options[3]}
    
    trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory.lower())
    
    try:
        tester = LSTMTester(args.model_path, device=args.device)
        tester.test(CONFIG_MAP[args.delay_config], trajectory_type, args.num_steps)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()