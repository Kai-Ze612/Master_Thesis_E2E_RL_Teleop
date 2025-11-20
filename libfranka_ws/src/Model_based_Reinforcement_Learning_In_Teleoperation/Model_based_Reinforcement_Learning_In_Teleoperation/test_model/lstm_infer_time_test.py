import time
import torch
import numpy as np
from collections import deque
import statistics

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    RNN_SEQUENCE_LENGTH,
    LSTM_MODEL_PATH
)

def benchmark():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    model = StateEstimator().to(device)
    try:
        checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint.get('state_estimator_state_dict', checkpoint))
    except:
        print("Using random weights.")

    model.eval()

    # --- APPLY OPTIMIZATIONS ---
    if device.type == 'cuda':
        model.half()
        print("Enabled FP16.")
        
    # JIT Trace
    dummy_input = torch.randn(1, RNN_SEQUENCE_LENGTH, N_JOINTS*2+1).to(device)
    if device.type == 'cuda':
        dummy_input = dummy_input.half()
        
    model = torch.jit.trace(model, dummy_input)
    print("Enabled JIT Tracing.")
    # ---------------------------

    buffer_len = RNN_SEQUENCE_LENGTH + 20
    leader_q_history = deque([np.random.randn(N_JOINTS).astype(np.float32) for _ in range(buffer_len)], maxlen=buffer_len)
    leader_qd_history = deque([np.random.randn(N_JOINTS).astype(np.float32) for _ in range(buffer_len)], maxlen=buffer_len)
    
    SEQ_LEN = RNN_SEQUENCE_LENGTH
    TEST_ITERATIONS = 1000
    
    # Warmup
    with torch.no_grad(): _ = model(dummy_input)
    torch.cuda.synchronize()

    # Store timings
    t_total = []

    print(f"\nStarting Optimized Benchmark...")

    for i in range(TEST_ITERATIONS):
        t0 = time.perf_counter()

        # Data Prep
        buffer_q = list(leader_q_history)[-SEQ_LEN:]
        buffer_qd = list(leader_qd_history)[-SEQ_LEN:]
        current_delay_scalar = 50.0
        
        batch_list = []
        for q, qd in zip(buffer_q, buffer_qd):
            vec = np.concatenate([q, qd, [current_delay_scalar]])
            batch_list.append(vec)
        
        raw_data = np.array(batch_list, dtype=np.float32).flatten()
        
        # Transfer & Cast
        full_seq_t = torch.tensor(raw_data, dtype=torch.float32).to(device).reshape(1, SEQ_LEN, -1)
        if device.type == 'cuda':
            full_seq_t = full_seq_t.half()
        
        # Inference
        with torch.no_grad():
            _ = model(full_seq_t)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        t3 = time.perf_counter()
        t_total.append((t3 - t0) * 1000) 

    print("-" * 60)
    print(f"{'TOTAL Latency':<20} | {statistics.mean(t_total):<10.4f} ms")
    print("-" * 60)
    
    if statistics.mean(t_total) > 4.0:
        print("\n[CRITICAL] Still too slow! Reduce RNN_HIDDEN_DIM in robot_config.py.")
    else:
        print("\n[OK] Speed is good for 250Hz.")

if __name__ == "__main__":
    benchmark()