import torch
import torch.nn as nn
import argparse
import sys
from typing import Dict, Tuple, List

from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    OBS_DIM,
    STATE_BUFFER_LENGTH,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    PPO_MLP_HIDDEN_DIMS,
    PPO_ROLLOUT_STEPS,
    PPO_BATCH_SIZE,
    PPO_NUM_EPOCHS,
    PREDICTION_LOSS_WEIGHT,
    PPO_LOSS_WEIGHT,
)

class SequenceHead(nn.Module):
    """
    Simulates a sequence-processing head (e.g., LSTM) for prediction.
    """
    def __init__(self, input_dim: int = OBS_DIM, hidden_dim: int = RNN_HIDDEN_DIM, output_dim: int = 14):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=RNN_NUM_LAYERS,
            batch_first=True
        )
        
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
    def __init__(self, input_dim: int = 33, output_dim: int = N_JOINTS):  # 19 (remote) + 14 (predicted)
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, PPO_MLP_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(PPO_MLP_HIDDEN_DIMS[0], PPO_MLP_HIDDEN_DIMS[1]),
            nn.ReLU()
        )
        
        self.head_a = nn.Linear(PPO_MLP_HIDDEN_DIMS[1], output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
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
# ADVANCED GPU MEMORY PROFILER WITH REALISTIC DATA FLOW
# ============================================================================

class AdvancedGPUMemoryProfiler:
    """
    A professional GPU memory profiler that accurately simulates:
    1. Realistic tensor allocation (NumPy on CPU, selective GPU loading)
    2. Gradient accumulation patterns
    3. Mini-batch processing across multiple epochs
    4. Tensor lifecycle (creation, usage, deletion)
    """
    
    def __init__(self, buffer_size: int, batch_size: int, num_epochs: int = PPO_NUM_EPOCHS):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.seq_model = None
        self.mlp_model = None
        self.optimizer = None
        
        # For tracking memory across operations
        self.memory_log: List[Dict] = []
    
    def _measure_memory(self, label: str, detailed: bool = False) -> float:
        """Measure current GPU allocated memory in GB."""
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        peak = torch.cuda.max_memory_allocated(self.device) / 1e9
        
        if detailed:
            print(f"  {label}:")
            print(f"    Allocated: {allocated:.4f} GB")
            print(f"    Reserved:  {reserved:.4f} GB")
            print(f"    Peak:      {peak:.4f} GB")
        else:
            print(f"  {label}: {allocated:.4f} GB (reserved: {reserved:.4f} GB, peak: {peak:.4f} GB)")
        
        return allocated
    
    def _clear_gpu(self):
        """Clears GPU cache and resets peak memory tracking."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
    
    def run_profiler(self, gpu_limit_gb: float) -> Dict:
        """Runs the complete memory diagnostic."""
        
        print("=" * 90)
        print("ADVANCED GPU MEMORY PROFILER FOR DEEP LEARNING WORKLOAD")
        print("=" * 90)
        print(f"\nConfiguration:")
        print(f"  Buffer Size (Transitions):  {self.buffer_size:,}")
        print(f"  Mini-Batch Size:            {self.batch_size}")
        print(f"  Epochs per Update:          {self.num_epochs}")
        print(f"  Number of Batches per Epoch: {(self.buffer_size + self.batch_size - 1) // self.batch_size}")
        print(f"\nSystem Info:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Target Limit for Safety: {gpu_limit_gb:.1f} GB")
        
        results = {
            'stages': {},
            'tensor_sizes': {},
            'status': 'UNKNOWN'
        }
        
        # --- STAGE 1: Model Initialization ---
        print("\n" + "-" * 90)
        print("STAGE 1: Model Parameters Allocation")
        print("-" * 90)
        
        self._clear_gpu()
        
        try:
            self.seq_model = SequenceHead().to(self.device)
            self.mlp_model = MLPTrainerNetwork().to(self.device)
            models_mem = self._measure_memory("Models loaded")
            results['stages']['models'] = models_mem
            
        except RuntimeError as e:
            print(f"  ✗ ERROR: Could not load models: {e}")
            results['status'] = 'ERROR_MODEL_LOAD'
            return results
        
        # --- STAGE 2A: CPU Buffer Allocation (Realistic NumPy Storage) ---
        print("\n" + "-" * 90)
        print("STAGE 2A: Rollout Buffer Allocation (CPU NumPy Storage - Realistic)")
        print("-" * 90)
        print(f"  Allocating {self.buffer_size:,} data entries on CPU...")
        
        try:
            import numpy as np
            
            # Simulate actual buffer content (all in NumPy on CPU)
            buffer_cpu = {
                'delayed_sequences': np.random.randn(self.buffer_size, STATE_BUFFER_LENGTH, OBS_DIM).astype(np.float32),
                'remote_states': np.random.randn(self.buffer_size, 19).astype(np.float32),
                'actions': np.random.randn(self.buffer_size, N_JOINTS).astype(np.float32),
                'old_log_probs': np.random.randn(self.buffer_size).astype(np.float32),
                'old_values': np.random.randn(self.buffer_size).astype(np.float32),
                'advantages': np.random.randn(self.buffer_size).astype(np.float32),
                'returns': np.random.randn(self.buffer_size).astype(np.float32),
                'predicted_targets': np.random.randn(self.buffer_size, 14).astype(np.float32),
                'true_targets': np.random.randn(self.buffer_size, 14).astype(np.float32),
            }
            
            # Calculate CPU RAM usage (for reference)
            cpu_ram_gb = sum(arr.nbytes for arr in buffer_cpu.values()) / 1e9
            print(f"  CPU RAM used (NumPy): {cpu_ram_gb:.4f} GB")
            results['stages']['cpu_buffer'] = cpu_ram_gb
            
        except Exception as e:
            print(f"  ✗ ERROR: Could not allocate CPU buffer: {e}")
            results['status'] = 'ERROR_CPU_BUFFER'
            return results
        
        # --- STAGE 2B: GPU Buffer Allocation (Full Dataset to GPU) ---
        print("\n" + "-" * 90)
        print("STAGE 2B: Rollout Buffer Transfer to GPU (Full Dataset)")
        print("-" * 90)
        print(f"  Converting all {self.buffer_size:,} entries to GPU tensors...")
        
        self._clear_gpu()
        
        try:
            # Convert all buffer data to GPU (this is what buffer.get() does)
            buffer_gpu = {
                k: torch.FloatTensor(v).to(self.device)
                for k, v in buffer_cpu.items()
            }
            
            buffer_gpu_mem = self._measure_memory("Full buffer on GPU", detailed=True)
            results['stages']['buffer_on_gpu'] = buffer_gpu_mem
            results['tensor_sizes'] = {k: v.shape for k, v in buffer_gpu.items()}
            
            # Keep reference to prevent garbage collection
            self.buffer_gpu = buffer_gpu
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OUT OF MEMORY: Cannot convert buffer to GPU.")
                print(f"     This is where your memory exhaustion occurs!")
                results['status'] = 'OUT_OF_MEMORY_BUFFER_TO_GPU'
                return results
            else:
                raise
        
        # --- STAGE 3: Mini-Batch Processing (Single Epoch Simulation) ---
        print("\n" + "-" * 90)
        print(f"STAGE 3: Mini-Batch Processing (First Epoch, Batch Size {self.batch_size})")
        print("-" * 90)
        
        self._clear_gpu()
        
        try:
            self.optimizer = torch.optim.Adam(
                list(self.seq_model.parameters()) + list(self.mlp_model.parameters()),
                lr=3e-4
            )
            
            num_batches = (self.buffer_size + self.batch_size - 1) // self.batch_size
            peak_training_mem = 0.0
            
            print(f"  Processing {num_batches} batches...\n")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.buffer_size)
                
                # Extract mini-batch from GPU buffer
                batch_seq = buffer_gpu['delayed_sequences'][start_idx:end_idx]
                batch_remote = buffer_gpu['remote_states'][start_idx:end_idx]
                batch_targets = buffer_gpu['true_targets'][start_idx:end_idx]
                batch_actions = buffer_gpu['actions'][start_idx:end_idx]
                batch_returns = buffer_gpu['returns'][start_idx:end_idx]
                
                # CRITICAL: Zero gradients BEFORE backward pass
                self.optimizer.zero_grad()
                
                # Forward pass
                predicted = self.seq_model(batch_seq)
                policy_input = torch.cat([predicted, batch_remote], dim=-1)
                output_a, _, output_b = self.mlp_model(policy_input)
                
                # Compute losses
                loss_pred = nn.MSELoss()(predicted, batch_targets)
                loss_policy = ((output_a - batch_actions) ** 2).mean()
                loss_value = ((output_b.squeeze() - batch_returns) ** 2).mean()
                
                total_loss = (PREDICTION_LOSS_WEIGHT * loss_pred +
                              PPO_LOSS_WEIGHT * (loss_policy + 0.5 * loss_value))
                
                # Backward pass
                total_loss.backward()
                
                # Optimizer step
                self.optimizer.step()
                
                # Measure memory
                current_mem = torch.cuda.memory_allocated(self.device) / 1e9
                peak_mem_so_far = torch.cuda.max_memory_allocated(self.device) / 1e9
                peak_training_mem = max(peak_training_mem, peak_mem_so_far)
                
                if (batch_idx + 1) % max(1, num_batches // 5) == 0 or batch_idx == 0:
                    print(f"    Batch {batch_idx + 1}/{num_batches}: "
                          f"Current={current_mem:.4f} GB, "
                          f"Peak={peak_mem_so_far:.4f} GB, "
                          f"Loss={total_loss.item():.4f}")
            
            results['stages']['peak_training_single_epoch'] = peak_training_mem
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OUT OF MEMORY: During mini-batch training.")
                results['status'] = 'OUT_OF_MEMORY_TRAINING'
                return results
            else:
                raise
        
        # --- STAGE 4: Multi-Epoch Simulation (Gradient Accumulation Check) ---
        print("\n" + "-" * 90)
        print(f"STAGE 4: Multi-Epoch Simulation ({min(self.num_epochs, 3)} epochs)")
        print("-" * 90)
        print(f"  Checking for gradient accumulation issues...\n")
        
        self._clear_gpu()
        
        try:
            # Reset model and optimizer
            self.seq_model.zero_grad()
            self.mlp_model.zero_grad()
            
            epochs_to_test = min(self.num_epochs, 3)  # Limit to 3 for safety
            peak_multi_epoch_mem = 0.0
            
            for epoch in range(epochs_to_test):
                num_batches = (self.buffer_size + self.batch_size - 1) // self.batch_size
                epoch_loss = 0.0
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, self.buffer_size)
                    
                    batch_seq = buffer_gpu['delayed_sequences'][start_idx:end_idx]
                    batch_remote = buffer_gpu['remote_states'][start_idx:end_idx]
                    batch_targets = buffer_gpu['true_targets'][start_idx:end_idx]
                    batch_actions = buffer_gpu['actions'][start_idx:end_idx]
                    batch_returns = buffer_gpu['returns'][start_idx:end_idx]
                    
                    # CRITICAL: Zero gradients before backward
                    self.optimizer.zero_grad()
                    
                    predicted = self.seq_model(batch_seq)
                    policy_input = torch.cat([predicted, batch_remote], dim=-1)
                    output_a, _, output_b = self.mlp_model(policy_input)
                    
                    loss_pred = nn.MSELoss()(predicted, batch_targets)
                    loss_policy = ((output_a - batch_actions) ** 2).mean()
                    loss_value = ((output_b.squeeze() - batch_returns) ** 2).mean()
                    
                    total_loss = (PREDICTION_LOSS_WEIGHT * loss_pred +
                                  PPO_LOSS_WEIGHT * (loss_policy + 0.5 * loss_value))
                    
                    total_loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += total_loss.item()
                
                current_mem = torch.cuda.memory_allocated(self.device) / 1e9
                peak_mem_so_far = torch.cuda.max_memory_allocated(self.device) / 1e9
                peak_multi_epoch_mem = max(peak_multi_epoch_mem, peak_mem_so_far)
                
                avg_loss = epoch_loss / num_batches
                print(f"    Epoch {epoch + 1}/{epochs_to_test}: "
                      f"Avg Loss={avg_loss:.4f}, "
                      f"Current Mem={current_mem:.4f} GB, "
                      f"Peak Mem={peak_mem_so_far:.4f} GB")
            
            results['stages']['peak_training_multi_epoch'] = peak_multi_epoch_mem
            results['status'] = 'SUCCESS'
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OUT OF MEMORY: During multi-epoch training.")
                results['status'] = 'OUT_OF_MEMORY_MULTI_EPOCH'
                return results
            else:
                raise
        
        return results


def print_detailed_assessment(buffer_size: int, batch_size: int, gpu_limit_gb: float, results: Dict):
    """Provides detailed safety recommendations based on profiling results."""
    
    print("\n" + "=" * 90)
    print(f"COMPREHENSIVE VRAM SAFETY ASSESSMENT (Target GPU Limit: {gpu_limit_gb:.1f} GB)")
    print("=" * 90)
    
    status = results.get('status', 'UNKNOWN')
    
    # Failure cases
    if 'OUT_OF_MEMORY' in status:
        print(f"\n✗ UNSAFE: Diagnostic halted due to VRAM exhaustion.\n")
        
        if status == 'OUT_OF_MEMORY_BUFFER_TO_GPU':
            print(f"ROOT CAUSE: **Buffer Transfer to GPU**")
            print(f"  The full buffer cannot fit in GPU memory when converted from NumPy.")
            print(f"  This is where your OOM occurs!")
            print(f"\nDiagnostic:")
            buffer_mem = results['stages'].get('buffer_on_gpu', 0)
            print(f"  Buffer on GPU would use: ~{buffer_mem:.3f} GB (incomplete)")
            print(f"  Available VRAM: {gpu_limit_gb:.1f} GB")
            print(f"\nRecommendations:")
            print(f"  1. **Reduce Buffer Size** from {buffer_size:,} to {buffer_size // 2:,}")
            print(f"  2. **Reduce OBS_DIM** from {OBS_DIM} to {OBS_DIM - 20} (remove redundant observations)")
            print(f"  3. **Reduce STATE_BUFFER_LENGTH** from {STATE_BUFFER_LENGTH} to {STATE_BUFFER_LENGTH // 2}")
            print(f"  4. Use **Gradient Checkpointing** in LSTM to trade compute for memory")
            
        elif status == 'OUT_OF_MEMORY_TRAINING':
            print(f"ROOT CAUSE: **Mini-Batch Backward Pass**")
            print(f"  Gradients and intermediate activations exceed VRAM during backward.")
            print(f"\nDiagnostic:")
            buffer_mem = results['stages'].get('buffer_on_gpu', 0)
            models_mem = results['stages'].get('models', 0)
            print(f"  Models: {models_mem:.3f} GB")
            print(f"  Buffer: {buffer_mem:.3f} GB")
            print(f"  Available for backward: {gpu_limit_gb - buffer_mem - models_mem:.3f} GB (insufficient)")
            print(f"\nRecommendations:")
            print(f"  1. **Reduce Batch Size** from {batch_size} to {batch_size // 2}")
            print(f"  2. **Reduce RNN_HIDDEN_DIM** from {RNN_HIDDEN_DIM} to {RNN_HIDDEN_DIM // 2}")
            print(f"  3. **Enable Gradient Checkpointing** to save memory during backward")
            
        elif status == 'OUT_OF_MEMORY_MULTI_EPOCH':
            print(f"ROOT CAUSE: **Gradient Accumulation Across Epochs**")
            print(f"  Memory grows over multiple epochs (possible missing zero_grad() calls).")
            print(f"\nDiagnostic:")
            print(f"  This suggests optimizer.zero_grad() may not be called properly.")
            print(f"\nRecommendations:")
            print(f"  1. **Verify zero_grad() is called** BEFORE each backward pass")
            print(f"  2. Check training loop in ppo_policy_network.py")
            print(f"  3. Ensure: optimizer.zero_grad() → loss.backward() → optimizer.step()")
        
        return
    
    if status != 'SUCCESS':
        print(f"\n? UNKNOWN: Profiler did not complete (Status: {status})")
        return
    
    # Success case: detailed breakdown
    print(f"\n✓ SUCCESS: Profiler completed all stages.\n")
    
    models_mem = results['stages'].get('models', 0)
    buffer_mem = results['stages'].get('buffer_on_gpu', 0)
    peak_single = results['stages'].get('peak_training_single_epoch', 0)
    peak_multi = results['stages'].get('peak_training_multi_epoch', 0)
    
    print(f"Memory Breakdown:")
    print(f"  Model Parameters:              {models_mem:.4f} GB")
    print(f"  Rollout Buffer (GPU):          {buffer_mem:.4f} GB")
    print(f"  Peak (Single Epoch Training):  {peak_single:.4f} GB")
    print(f"  Peak (Multi-Epoch Training):   {peak_multi:.4f} GB")
    print(f"  Target GPU Limit:              {gpu_limit_gb:.4f} GB")
    
    print(f"\nTensor Sizes in Buffer:")
    for key, shape in results['tensor_sizes'].items():
        size_gb = (torch.randn(shape, dtype=torch.float32).nbytes / 1e9)
        print(f"  {key:25s}: {str(shape):30s} ({size_gb:.6f} GB)")
    
    utilization_single = (peak_single / gpu_limit_gb) * 100
    utilization_multi = (peak_multi / gpu_limit_gb) * 100
    
    print(f"\nUtilization Analysis:")
    print(f"  Single Epoch Peak:  {utilization_single:.1f}% of {gpu_limit_gb:.1f} GB")
    print(f"  Multi-Epoch Peak:   {utilization_multi:.1f}% of {gpu_limit_gb:.1f} GB")
    
    print(f"\nSafety Classification:")
    
    if utilization_multi < 75:
        recommendation = "SAFE"
        symbol = "✓"
        print(f"  {symbol} SAFE: {utilization_multi:.1f}% utilization (comfortable margin).")
        print(f"    You can safely increase buffer size or batch size if needed.")
    elif utilization_multi < 90:
        recommendation = "CAUTION"
        symbol = "⚠"
        print(f"  {symbol} CAUTION: {utilization_multi:.1f}% utilization (tight margin).")
        print(f"    Monitor closely during training. Avoid other GPU tasks.")
    else:
        recommendation = "RISKY/UNSAFE"
        symbol = "✗"
        print(f"  {symbol} RISKY/UNSAFE: {utilization_multi:.1f}% utilization (high OOM risk).")
        print(f"    You will likely experience out-of-memory errors.")
    
    print(f"\nRecommended Actions:")
    if recommendation == 'SAFE':
        print(f"  Configuration is suitable for production training.")
        print(f"  Consider profiling with larger batch sizes to find optimal throughput.")
    elif recommendation == 'CAUTION':
        print(f"  1. Run training with this configuration, but monitor GPU memory")
        print(f"  2. If OOM occurs, reduce batch size to {batch_size // 2}")
        print(f"  3. Alternatively, reduce buffer size to {buffer_size // 2:,}")
    else:
        print(f"  1. Reduce Batch Size: {batch_size} → {max(16, batch_size // 2)}")
        print(f"  2. Reduce Buffer Size: {buffer_size:,} → {buffer_size // 2:,}")
        print(f"  3. Enable Gradient Checkpointing in SequenceHead")
        print(f"  4. Reduce OBS_DIM or STATE_BUFFER_LENGTH in config")
    
    print(f"\nCritical Checklist:")
    print(f"  ✓ Verify optimizer.zero_grad() is called BEFORE loss.backward()")
    print(f"  ✓ Verify buffer.get() is called only once per epoch cycle")
    print(f"  ✓ Verify no gradient accumulation is happening unintentionally")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced GPU memory profiler with realistic data flow simulation."
    )
    parser.add_argument('--gpulimit', type=float, default=4.0,
                        help='Maximum VRAM (GB) for safety assessment (default: 4.0)')
    parser.add_argument('--buffersize', type=int, default=PPO_ROLLOUT_STEPS,
                        help=f'Buffer size in transitions (default: {PPO_ROLLOUT_STEPS:,})')
    parser.add_argument('--batchsize', type=int, default=PPO_BATCH_SIZE,
                        help=f'Mini-batch size (default: {PPO_BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=PPO_NUM_EPOCHS,
                        help=f'Number of epochs to simulate (default: {PPO_NUM_EPOCHS})')

    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not detected. This profiler requires a functional GPU.")
        sys.exit(1)
    
    try:
        profiler = AdvancedGPUMemoryProfiler(
            buffer_size=args.buffersize,
            batch_size=args.batchsize,
            num_epochs=args.epochs
        )
        results = profiler.run_profiler(args.gpulimit)
        print_detailed_assessment(args.buffersize, args.batchsize, args.gpulimit, results)
        
        print("\n" + "=" * 90)
        print("PROFILER COMPLETE")
        print("=" * 90)
        
    except Exception as e:
        print(f"\nRUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()