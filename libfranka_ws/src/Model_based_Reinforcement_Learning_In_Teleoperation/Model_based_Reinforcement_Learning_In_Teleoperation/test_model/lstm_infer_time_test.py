import torch
import torch.nn as nn
import torch.quantization
import time
import numpy as np
import os
import sys

# --- IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

# --- WRAPPER FOR BENCHMARKING ---
class AutoregressiveLoop(nn.Module):
    def __init__(self, state_estimator, n_joints, target_delta_scale, delay_input_norm_factor, dt):
        super().__init__()
        self.state_estimator = state_estimator
        self.n_joints = n_joints
        self.target_delta_scale = target_delta_scale
        self.dt_norm = dt / delay_input_norm_factor
        
    def forward(self, current_q, current_qd, hidden_h, hidden_c, start_delay, steps: int):
        current_delay = start_delay
        for _ in range(steps):
            delay_t = current_delay.view(1, 1, 1)
            step_input = torch.cat([current_q, current_qd, delay_t], dim=2)
            
            # Forward step
            lstm_out, (hidden_h, hidden_c) = self.state_estimator.lstm(step_input, (hidden_h, hidden_c))
            residual = self.state_estimator.fc(lstm_out[:, -1, :])
            
            pred_residual = residual * self.target_delta_scale
            pred_residual = torch.clamp(pred_residual, -0.2, 0.2)
            
            current_q = current_q + pred_residual[:, :self.n_joints].unsqueeze(1)
            current_qd = current_qd + pred_residual[:, self.n_joints:].unsqueeze(1)
            current_delay = current_delay + self.dt_norm
            
        return current_q, current_qd

def run_quantization_test():
    device = torch.device('cpu')
    torch.set_num_threads(1)
    
    print(f"==================================================")
    print(f"--- QUANTIZATION (INT8) SPEED TEST ---")
    print(f"==================================================")
    print(f"Original Model: Hidden={cfg.RNN_HIDDEN_DIM}, Layers={cfg.RNN_NUM_LAYERS}")
    print(f"Steps: 100 (Worst Case)")
    
    # 1. Load Float32 Model
    float_model = StateEstimator().to(device)
    float_model.eval()
    
    # 2. Quantize to INT8
    print("\n[1] Quantizing Model (Float32 -> Int8)...")
    # PyTorch Dynamic Quantization targets Linear and LSTM layers
    quantized_model = torch.quantization.quantize_dynamic(
        float_model, 
        {nn.LSTM, nn.Linear}, 
        dtype=torch.qint8
    )
    print("    Model size reduced significantly.")

    # 3. Create Wrapper
    jit_module = AutoregressiveLoop(
        quantized_model, # Use the quantized estimator
        cfg.N_JOINTS, 
        cfg.TARGET_DELTA_SCALE, 
        cfg.DELAY_INPUT_NORM_FACTOR, 
        1.0/cfg.DEFAULT_CONTROL_FREQ
    ).to(device)

    # 4. Prepare Inputs
    dummy_q = torch.zeros(1, 1, 7).to(device)
    dummy_qd = torch.zeros(1, 1, 7).to(device)
    dummy_h = torch.zeros(cfg.RNN_NUM_LAYERS, 1, cfg.RNN_HIDDEN_DIM).to(device)
    dummy_c = torch.zeros(cfg.RNN_NUM_LAYERS, 1, cfg.RNN_HIDDEN_DIM).to(device)
    dummy_delay = torch.tensor([0.1]).to(device)

    # 5. Benchmark
    print("\n[2] Running Benchmark (100 iters)...")
    
    # Warmup
    for _ in range(10): jit_module(dummy_q, dummy_qd, dummy_h, dummy_c, dummy_delay, 100)
    
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        jit_module(dummy_q, dummy_qd, dummy_h, dummy_c, dummy_delay, 100)
        times.append((time.perf_counter() - t0) * 1000)
        
    avg_time = np.mean(times)
    
    print(f"\n--- RESULTS ---")
    print(f"INT8 Time: {avg_time:.3f} ms")
    
    if avg_time > 5.0:
        print(f"[FAIL] INT8 is still too slow ({avg_time:.2f}ms).")
        print("VERDICT: You MUST reduce the model size (Hidden=64). Quantization is not enough.")
    else:
        print(f"[PASS] INT8 saved the day! ({avg_time:.2f}ms)")

if __name__ == "__main__":
    run_quantization_test()