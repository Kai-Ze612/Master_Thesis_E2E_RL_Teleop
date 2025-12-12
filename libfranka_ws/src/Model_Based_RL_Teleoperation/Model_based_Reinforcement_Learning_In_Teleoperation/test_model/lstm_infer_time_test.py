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

def run_single_step_benchmark():
    device = torch.device('cpu') 
    torch.set_num_threads(1)
    
    print(f"==================================================")
    print(f"--- SINGLE STEP LSTM INFERENCE TEST (CPU) ---")
    print(f"==================================================")
    print(f"Model: Hidden={cfg.RNN_HIDDEN_DIM}, Layers={cfg.RNN_NUM_LAYERS}")
    
    # 1. SETUP MODEL (INT8)
    # We use INT8 because your previous test showed it was 4x faster
    model = StateEstimator().to(device)
    model.eval()
    model = torch.quantization.quantize_dynamic(
        model, 
        {nn.LSTM, nn.Linear}, 
        dtype=torch.qint8
    )

    # 2. DUMMY DATA (Batch=1, Seq=1)
    # Input: [Position(7) + Velocity(7) + Delay(1)]
    dummy_input = torch.zeros(1, 1, 15).to(device) 
    dummy_h = torch.zeros(cfg.RNN_NUM_LAYERS, 1, cfg.RNN_HIDDEN_DIM).to(device)
    dummy_c = torch.zeros(cfg.RNN_NUM_LAYERS, 1, cfg.RNN_HIDDEN_DIM).to(device)

    # 3. BENCHMARK
    print(f"\n[Running Benchmark: 1000 Iterations]...")

    # Warmup
    for _ in range(100): 
        model.forward_step(dummy_input, (dummy_h, dummy_c))
    
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        
        # --- SINGLE STEP INFERENCE ---
        with torch.no_grad():
            pred, (new_h, new_c) = model.forward_step(dummy_input, (dummy_h, dummy_c))
        # -----------------------------
        
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000) # ms
    
    avg_ms = np.mean(times)
    std_ms = np.std(times)
    
    # 4. ANALYSIS
    control_limit_ms = (1.0 / cfg.DEFAULT_CONTROL_FREQ) * 1000 # 5ms
    max_bridgeable_steps = int(control_limit_ms / avg_ms)
    max_delay_ms = max_bridgeable_steps * (1000 / cfg.DEFAULT_CONTROL_FREQ)

    print(f"\n--- RESULTS ---")
    print(f"Single Step Latency: {avg_ms:.4f} ms ± {std_ms:.4f}")
    print(f"Control Cycle Limit: {control_limit_ms:.2f} ms")
    print(f"\n--- FEASIBILITY ---")
    print(f"Max steps you can bridge in one cycle: {max_bridgeable_steps} steps")
    print(f"Max delay you can handle synchronously:  {max_delay_ms:.1f} ms")
    
    if avg_ms < control_limit_ms:
        print(f"[PASS] Single step is fast enough.")
    else:
        print(f"[FAIL] Single step is too slow ({avg_ms} > {control_limit_ms}). Reduce Model Size.")

if __name__ == "__main__":
    run_single_step_benchmark()