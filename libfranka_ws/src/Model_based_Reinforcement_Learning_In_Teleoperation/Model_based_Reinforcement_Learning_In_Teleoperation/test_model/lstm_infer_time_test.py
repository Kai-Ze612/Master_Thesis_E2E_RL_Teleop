import torch
import torch.nn as nn
import time
import numpy as np
import os
import sys

# Update path to find your modules
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

# --- JIT WRAPPER MODULE ---
class AutoregressiveLoop(nn.Module):
    def __init__(self, state_estimator, n_joints, target_delta_scale, delay_input_norm_factor, dt):
        super().__init__()
        self.state_estimator = state_estimator
        self.n_joints = n_joints
        self.target_delta_scale = target_delta_scale
        self.dt_norm = dt / delay_input_norm_factor
        
    def forward(self, current_q, current_qd, hidden_h, hidden_c, start_delay, steps: int):
        """
        This entire loop will be compiled into C++/CUDA code.
        """
        # We need to decompose the LSTM hidden state tuple for JIT
        current_delay = start_delay
        
        # Lists to store outputs if needed, or just keep the last state
        # For control, we usually just need the final state or the sequence
        
        for _ in range(steps):
            # 1. Create Input Tensor on device immediately
            # Shape: (1, 1, 1) -> view -> (1, 1, 1)
            delay_t = current_delay.view(1, 1, 1)
            
            # Cat: (1, 1, 14) + (1, 1, 1) -> (1, 1, 15)
            step_input = torch.cat([current_q, current_qd, delay_t], dim=2)
            
            # 2. Forward Step (Manually unrolled for JIT compatibility if needed)
            # calling self.state_estimator.forward_step
            lstm_out, (hidden_h, hidden_c) = self.state_estimator.lstm(step_input, (hidden_h, hidden_c))
            residual = self.state_estimator.fc(lstm_out[:, -1, :])
            
            # 3. Physics Update
            pred_residual = residual * self.target_delta_scale
            pred_residual = torch.clamp(pred_residual, -0.2, 0.2)
            
            current_q = current_q + pred_residual[:, :self.n_joints].unsqueeze(1)
            current_qd = current_qd + pred_residual[:, self.n_joints:].unsqueeze(1)
            current_delay = current_delay + self.dt_norm
            
        return current_q, current_qd, (hidden_h, hidden_c)

def run_test():
    device = torch.device('cuda') # Test on GPU to see improvement
    print(f"Testing JIT Compilation on {device}...")
    
    # Load Weights
    raw_model = StateEstimator().to(device)
    # (Load your weights here if needed, skipping for speed test)
    raw_model.eval()
    
    # Create Wrapper
    jit_module = AutoregressiveLoop(
        raw_model, 
        cfg.N_JOINTS, 
        cfg.TARGET_DELTA_SCALE, 
        cfg.DELAY_INPUT_NORM_FACTOR, 
        1.0/cfg.DEFAULT_CONTROL_FREQ
    ).to(device)
    
    # -------------------------------------------------
    # COMPILE (The Magic Step)
    # -------------------------------------------------
    # Create dummy inputs for tracing
    dummy_q = torch.zeros(1, 1, 7).to(device)
    dummy_qd = torch.zeros(1, 1, 7).to(device)
    dummy_h = torch.zeros(3, 1, 256).to(device) # Layers=3, Batch=1, Hidden=256
    dummy_c = torch.zeros(3, 1, 256).to(device)
    dummy_delay = torch.tensor([0.1]).to(device)
    dummy_steps = torch.tensor(100) # JIT prefers tensors or constant ints
    
    print("Compiling via TorchScript...")
    # We use script() instead of trace() for loops with variable steps
    scripted_model = torch.jit.script(jit_module)
    print("Compilation Complete.")
    
    # -------------------------------------------------
    # SPEED TEST
    # -------------------------------------------------
    # Warmup
    for _ in range(50):
        scripted_model(dummy_q, dummy_qd, dummy_h, dummy_c, dummy_delay, 100)
    torch.cuda.synchronize()
    
    print("Running Benchmark (1000 iters)...")
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = scripted_model(dummy_q, dummy_qd, dummy_h, dummy_c, dummy_delay, 100)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        
    print(f"Avg JIT Time: {np.mean(times):.3f} ms")
    print(f"Max JIT Time: {np.max(times):.3f} ms")

if __name__ == "__main__":
    run_test()