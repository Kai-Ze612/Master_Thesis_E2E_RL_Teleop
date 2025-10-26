import torch
import torch.nn as nn
import argparse
import sys
from typing import Dict, Tuple

from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    OBS_DIM,
    STATE_BUFFER_LENGTH,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    PPO_MLP_HIDDEN_DIMS,
    PPO_ROLLOUT_STEPS_DEFAULT,
    PPO_BATCH_SIZE_DEFAULT,
    PPO_NUM_EPOCHS,
    PREDICTION_LOSS_WEIGHT,
    PPO_LOSS_WEIGHT
)

class SequenceHead(nn.Module):
    """
    Simulates a sequence-processing head (e.g., LSTM) for prediction.
    """
    def __init__(self, input_dim: int = OBS_DIM, hidden_dim: int = RNN_HIDDEN_DIM, output_dim: int = 14):
        super().__init__()
        
        # LSTM configuration
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=RNN_NUM_LAYERS,
            batch_first=True
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            predictions: (batch_size, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        predictions = self.prediction_head(last_hidden)
        return predictions


class MLPTrainerNetwork(nn.Module):
    """
    Simulates an Actor-Critic/Policy network (e.g., PPO) with MLP structure.
    """
    def __init__(self, input_dim: int = 28, output_dim: int = N_JOINTS):
        super().__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, PPO_MLP_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(PPO_MLP_HIDDEN_DIMS[0], PPO_MLP_HIDDEN_DIMS[1]),
            nn.ReLU()
        )
        
        # Output Head 1 (e.g., action mean)
        self.head_a = nn.Linear(PPO_MLP_HIDDEN_DIMS[1], output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
        # Output Head 2 (e.g., value function)
        self.head_b = nn.Sequential(
            nn.Linear(PPO_MLP_HIDDEN_DIMS[1], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        output_a = self.head_a(shared_features)
        output_std = torch.exp(self.log_std.unsqueeze(0).expand(x.shape[0], -1))
        output_b = self.head_b(shared_features)
        
        return output_a, output_std, output_b


# ============================================================================
# GPU MEMORY PROFILER CLASS
# ============================================================================

class GPUMemoryProfiler:
    """A utility class to profile GPU memory usage for large-scale training simulation."""
    
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.seq_model = None
        self.mlp_model = None
        self.optimizer = None
    
    def _measure_memory(self, label: str) -> float:
        """Measure current GPU allocated memory in GB."""
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        print(f"  {label}: {allocated:.3f} GB")
        return allocated
    
    def _clear_gpu(self):
        """Clears GPU cache."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
    
    def run_profiler(self, gpu_limit_gb: float) -> Dict[str, float]:
        """Runs the complete memory diagnostic and returns results."""
        
        print("=" * 80)
        print("GPU MEMORY PROFILER FOR DEEP LEARNING WORKLOAD")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Buffer Size (Transitions): {self.buffer_size:,}")
        print(f"  Mini-Batch Size:           {self.batch_size}")
        print(f"  Epochs per Update:         {NUM_EPOCHS_DEFAULT}")
        print(f"\nSystem Info:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Target Limit for Safety: {gpu_limit_gb:.1f} GB")
        
        results = {}
        
        # --- STAGE 1: Model Initialization ---
        print("\n" + "-" * 80)
        print("STAGE 1: Model Parameters Allocation")
        print("-" * 80)
        
        self._clear_gpu()
        
        try:
            self.seq_model = SequenceHead().to(self.device)
            self.mlp_model = MLPTrainerNetwork().to(self.device)
            models_mem = self._measure_memory("Models loaded")
            results['models_only'] = models_mem
            
        except RuntimeError as e:
            print(f"  ✗ ERROR: Could not load models. Check model size and VRAM: {e}")
            return results
        
        # --- STAGE 2: Rollout Buffer Allocation ---
        print("\n" + "-" * 80)
        print("STAGE 2: Rollout Buffer Allocation (Full Dataset on GPU)")
        print("-" * 80)
        print(f"  Allocating {self.buffer_size:,} data entries...")
        
        self._clear_gpu()
        
        try:
            # The largest tensor defines the bulk of the memory usage
            # Simulating all necessary data fields for training
            data_seq = torch.randn(
                self.buffer_size, STATE_BUFFER_LENGTH, OBS_DIM,
                dtype=torch.float32, device=self.device
            )
            data_mlp = torch.randn(
                self.buffer_size, 14,
                dtype=torch.float32, device=self.device
            )
            data_action = torch.randn(
                self.buffer_size, N_JOINTS,
                dtype=torch.float32, device=self.device
            )
            data_targets = torch.randn(
                self.buffer_size, 14,
                dtype=torch.float32, device=self.device
            )
            data_returns = torch.randn(
                self.buffer_size, 1,
                dtype=torch.float32, device=self.device
            )
            
            buffer_with_models = self._measure_memory("Models + Buffer allocated")
            results['buffer_only'] = buffer_with_models - results['models_only']
            results['total_before_training'] = buffer_with_models
            
            # Keep references to prevent garbage collection
            self.data_seq = data_seq
            self.data_mlp = data_mlp
            self.data_action = data_action
            self.data_targets = data_targets
            self.data_returns = data_returns
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OUT OF MEMORY: Cannot allocate {self.buffer_size:,} data entries.")
                print(f"  This configuration likely EXCEEDS the available VRAM.")
                results['status'] = 'OUT_OF_MEMORY_BUFFER'
                return results
            else:
                raise
        
        # --- STAGE 3: Backward Pass (Gradient Computation Simulation) ---
        print("\n" + "-" * 80)
        print(f"STAGE 3: Peak Training Memory (Backward Pass on batch size {self.batch_size})")
        print("-" * 80)
        
        self._clear_gpu()
        
        try:
            # Initialize Optimizer
            self.optimizer = torch.optim.Adam(
                list(self.seq_model.parameters()) + list(self.mlp_model.parameters()),
                lr=3e-4
            )
            
            # Use mini-batch from the allocated data
            batch_seq = self.data_seq[:self.batch_size]
            batch_mlp = self.data_mlp[:self.batch_size]
            batch_targets = self.data_targets[:self.batch_size]
            batch_actions = self.data_action[:self.batch_size]
            batch_returns = self.data_returns[:self.batch_size]
            
            # Forward pass
            predicted = self.seq_model(batch_seq)
            policy_input = torch.cat([predicted, batch_mlp], dim=-1)
            output_a, _, output_b = self.mlp_model(policy_input)
            
            # Loss computation
            loss_a = nn.MSELoss()(predicted, batch_targets)
            loss_b_policy = ((output_a - batch_actions) ** 2).mean()
            loss_b_value = ((output_b.squeeze() - batch_returns.squeeze()) ** 2).mean()
            
            total_loss = (LOSS_WEIGHT_A * loss_a +
                          LOSS_WEIGHT_B * (loss_b_policy + 0.5 * loss_b_value))
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self._measure_memory("After backward + optimizer step")
            results['peak_backward'] = torch.cuda.max_memory_allocated(self.device) / 1e9
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OUT OF MEMORY: Cannot compute gradients on batch of {self.batch_size}.")
                results['status'] = 'OUT_OF_MEMORY_TRAINING'
                return results
            else:
                raise
        
        results['status'] = 'SUCCESS'
        return results


def print_safety_assessment(buffer_size: int, batch_size: int, gpu_limit_gb: float, results: Dict[str, float]):
    """Provides safety recommendations based on the profiling results."""
    
    print("\n" + "=" * 80)
    print(f"VRAM SAFETY ASSESSMENT (Target GPU Limit: {gpu_limit_gb:.1f} GB)")
    print("=" * 80)
    
    # Handle failure statuses
    if 'OUT_OF_MEMORY' in results.get('status', ''):
        print(f"\n✗ UNSAFE: Diagnostic halted due to VRAM exhaustion.")
        if results['status'] == 'OUT_OF_MEMORY_BUFFER':
            print(f"   The primary issue is the **Rollout Buffer Size ({buffer_size:,})**.")
            print(f"   Recommendation: **Significantly reduce the Buffer Size** to below {buffer_size // 2:,}.")
        elif results['status'] == 'OUT_OF_MEMORY_TRAINING':
            print(f"   The primary issue is **Peak Training Memory** on batch size {batch_size}.")
            print(f"   Recommendation: **Reduce the Mini-Batch Size** (e.g., to {batch_size // 2}) or reduce Buffer Size.")
        return
    
    if results.get('status') != 'SUCCESS':
        print(f"\n? UNKNOWN: Diagnostic did not complete successfully.")
        return
    
    # Successful diagnostic output
    peak_mem = results.get('peak_backward', 0)
    buffer_mem = results.get('buffer_only', 0)
    models_mem = results.get('models_only', 0)
    
    utilization = (peak_mem / gpu_limit_gb) * 100
    
    print(f"\nMemory Breakdown:")
    print(f"  Model Weights:        {models_mem:.3f} GB")
    print(f"  Rollout Buffer (Max): {buffer_mem:.3f} GB")
    print(f"  Peak Training Usage:  {peak_mem:.3f} GB ({utilization:.1f}%)")
    print(f"  Remaining Headroom:   {gpu_limit_gb - peak_mem:.3f} GB")
    
    print(f"\nSafety Classification:")
    if utilization < 75:
        recommendation = "SAFE"
        print(f"  ✓ SAFE: {utilization:.1f}% utilization (comfortable margin).")
    elif utilization < 90:
        recommendation = "CAUTION"
        print(f"  ⚠ CAUTION: {utilization:.1f}% utilization (tight margin, monitor closely).")
    else:
        recommendation = "RISKY/UNSAFE"
        print(f"  ✗ RISKY/UNSAFE: {utilization:.1f}% utilization (high risk of OOM errors).")
    
    print(f"\nRecommended Actions:")
    if recommendation == 'CAUTION':
        print(f"  Consider reducing the Buffer Size or Mini-Batch Size for stability.")
    elif recommendation == 'RISKY/UNSAFE':
        print(f"  1. **Reduce Buffer Size:** Try {buffer_size // 2:,} or less.")
        print(f"  2. **Reduce Mini-Batch Size:** Try {batch_size // 2} or less.")


def main():
    parser = argparse.ArgumentParser(
        description="A professional utility for profiling GPU VRAM usage for large deep learning models."
    )
    # Target GPU limit is now an explicit argument
    parser.add_argument('--gpulimit', type=float, default=4.0,
                        help=f'The maximum VRAM (in GB) to consider for the safety assessment (default: 4.0).')
    parser.add_argument('--buffersize', type=int, default=ROLLOUT_STEPS_DEFAULT,
                        help=f'The total number of data transitions to allocate (default: {ROLLOUT_STEPS_DEFAULT:,}).')
    parser.add_argument('--batchsize', type=int, default=BATCH_SIZE_DEFAULT,
                        help=f'The mini-batch size used for gradient computation (default: {BATCH_SIZE_DEFAULT}).')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not detected. This profiler requires a functional GPU.")
        sys.exit(1)
    
    try:
        profiler = GPUMemoryProfiler(
            buffer_size=args.buffersize,
            batch_size=args.batchsize
        )
        results = profiler.run_profiler(args.gpulimit)
        print_safety_assessment(args.buffersize, args.batchsize, args.gpulimit, results)
        
        print("\n" + "=" * 80)
        print("PROFILER COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nRUNTIME ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()