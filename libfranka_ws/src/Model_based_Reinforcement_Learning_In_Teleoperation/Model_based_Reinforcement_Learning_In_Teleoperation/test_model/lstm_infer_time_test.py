import torch
import time
import numpy as np
import os
import sys

# --- IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator, Actor
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu') # Uncomment to force CPU test (often faster for small batches!)
print(f"Using device: {DEVICE}")

CONTROL_FREQ = 200  # Hz
BUDGET_MS = (1.0 / CONTROL_FREQ) * 1000

# Worst case scenario: How many steps do we predict if delay is max?
# Example: 500ms delay @ 200Hz = 100 steps
MAX_DELAY_STEPS = 50
ITERATIONS = 1000  # How many times to repeat the test

def run_timing_test():
    print(f"==================================================")
    print(f"--- INFERENCE LATENCY STRESS TEST ---")
    print(f"==================================================")
    print(f"Device:           {DEVICE}")
    print(f"Control Freq:     {CONTROL_FREQ} Hz")
    print(f"Time Budget:      {BUDGET_MS:.2f} ms")
    print(f"Simulating Delay: {MAX_DELAY_STEPS} steps (Worst Case)")
    print(f"==================================================")

    # 1. Initialize Models (Random weights are fine for speed testing)
    print("Initializing models...")
    lstm = StateEstimator().to(DEVICE)
    actor = Actor(state_dim=cfg.OBS_DIM).to(DEVICE)
    lstm.eval()
    actor.eval()

    # 2. Prepare Dummy Data
    # Batch size 1, Sequence length 150
    seq_len = cfg.RNN_SEQUENCE_LENGTH
    # Input dim: q(7) + qd(7) + delay(1) = 15
    dummy_input_seq = torch.randn(1, seq_len, 15).to(DEVICE)
    
    # Pre-calculate constants
    dt_norm = (1.0 / CONTROL_FREQ) / cfg.DELAY_INPUT_NORM_FACTOR
    
    # 3. Warmup (Crucial for CUDA)
    print("Warming up JIT/CUDA...")
    with torch.no_grad():
        for _ in range(50):
            _, hidden = lstm.lstm(dummy_input_seq)
            curr_input = dummy_input_seq[:, -1:, :]
            for _ in range(10): # Short loop
                _, hidden = lstm.forward_step(curr_input, hidden)
            _ = actor(torch.randn(1, cfg.OBS_DIM).to(DEVICE))
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    # 4. The Critical Path Loop
    # This matches `agent.py` logic exactly
    times = []
    
    print(f"Running {ITERATIONS} iterations...")
    
    for i in range(ITERATIONS):
        t_start = time.perf_counter()
        
        with torch.no_grad():
            # --- BLOCK A: LSTM INITIALIZATION ---
            # Run history sequence
            _, hidden_state = lstm.lstm(dummy_input_seq)
            
            # Setup for AR loop
            # (In real code we slice, here we just take last for speed test consistency)
            current_input = dummy_input_seq[:, -1:, :].clone()
            
            # --- BLOCK B: AUTOREGRESSIVE LOOP (The Bottleneck) ---
            # This is the heavy part: running the LSTM cell N times
            for _ in range(MAX_DELAY_STEPS):
                # 1. Forward
                pred_delta, hidden_state = lstm.forward_step(current_input, hidden_state)
                
                # 2. Math Overhead (Scale/Clamp/Add)
                # We simulate the cost of tensor operations
                last_known_state = current_input[:, :, :14]
                predicted_next_state = last_known_state + (pred_delta.unsqueeze(1) * 0.1)
                
                # 3. Next Input Construction
                # (Simulating concatenation cost)
                next_delay = torch.tensor([[[0.5]]], device=DEVICE) 
                current_input = torch.cat([predicted_next_state, next_delay], dim=2)

            # --- BLOCK C: ACTOR INFERENCE ---
            # Once LSTM finishes, we run Actor once
            # Construct dummy observation 113D
            dummy_obs = torch.randn(1, cfg.OBS_DIM).to(DEVICE)
            _ = actor(dummy_obs)

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
            
        t_end = time.perf_counter()
        times.append((t_end - t_start) * 1000.0) # Convert to ms

    # 5. Analysis
    avg_time = np.mean(times)
    max_time = np.max(times)
    min_time = np.min(times)
    p99_time = np.percentile(times, 99)

    print(f"\n--- RESULTS ---")
    print(f"Avg Time:  {avg_time:.3f} ms")
    print(f"Min Time:  {min_time:.3f} ms")
    print(f"Max Time:  {max_time:.3f} ms")
    print(f"99%ile:   {p99_time:.3f} ms")
    
    print(f"--------------------------------------------------")
    if p99_time > BUDGET_MS:
        print(f"[FAIL] SPEED TOO SLOW. {p99_time:.2f}ms > {BUDGET_MS:.2f}ms Limit")
        print("Recommendation: Use CPU instead of GPU, or reduce Control Freq.")
    else:
        margin = BUDGET_MS - p99_time
        print(f"[PASS] System is Real-Time Safe. Margin: {margin:.2f} ms")

if __name__ == "__main__":
    run_timing_test()