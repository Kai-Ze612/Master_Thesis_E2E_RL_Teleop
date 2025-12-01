"""
Comprehensive Validation Script for 7-Joint LSTM Prediction.
Aligns strictly with the real-time deployment logic in `agent.py`.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import deque

# --- IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import (
    LocalRobotSimulator, TrajectoryType
)
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

# --- CONFIG ---
TEST_CONFIG = ExperimentConfig.FULL_RANGE_COVER 
TEST_DURATION_SEC = 60.0  # Shorter duration for detailed view, or 60.0 for full test

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 7-JOINT LSTM VALIDATION (Deployment Logic) ---")
    print(f"Config: {TEST_CONFIG.name}")
    print(f"Scale: {cfg.TARGET_DELTA_SCALE}, Norm: {cfg.DELAY_INPUT_NORM_FACTOR}")

    # 1. Load Model
    model_path = cfg.LSTM_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = cfg.LSTM_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    model = StateEstimator().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_estimator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_estimator_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.")

    # 2. Setup Components
    dt = 1.0 / cfg.DEFAULT_CONTROL_FREQ
    total_steps = int(TEST_DURATION_SEC / dt)
    
    # Simulator (Leader)
    local_sim = LocalRobotSimulator(
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_params=False
    )
    local_sim.reset()
    
    # Delay Simulator
    delay_sim = DelaySimulator(cfg.DEFAULT_CONTROL_FREQ, TEST_CONFIG, seed=42)
    
    # 3. Deployment-Like Loop
    # We will simulate the AgentNode's behavior step-by-step
    
    # Buffers (same as AgentNode)
    leader_q_hist = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
    leader_qd_hist = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
    
    # Fill buffers initially (Warmup logic)
    q_start, qd_start, _, _, _, _ = local_sim.step()
    for _ in range(cfg.RNN_SEQUENCE_LENGTH + 50):
        leader_q_hist.append(q_start)
        leader_qd_hist.append(np.zeros_like(qd_start)) # Zero velocity start
        
    # Metrics
    history_truth_q = []
    history_pred_q = []
    history_error = []
    history_delay = []
    
    print("Running deployment simulation...")
    dt_norm = dt / cfg.DELAY_INPUT_NORM_FACTOR
    
    for t in tqdm(range(total_steps)):
        # --- A. Environment Step (The "Real World") ---
        # Leader moves
        q_real, qd_real, _, _, _, _ = local_sim.step()
        
        # Determine delay for this moment
        delay_steps = delay_sim.get_observation_delay_steps(len(leader_q_hist))
        
        # Update Agent's buffer with the "delayed" packet arrival
        # In this simplified sim, we just push the current state to the buffer
        # The delay logic happens during Retrieval
        leader_q_hist.append(q_real)
        leader_qd_hist.append(qd_real)
        
        # --- B. Agent Logic (Prediction) ---
        
        # 1. Prepare Input Sequence
        # Get data from (Current - Delay)
        # Note: In real deque, [-1] is most recent arrival.
        # But here we simulate delay by looking back into the deque relative to "now"
        
        # Calculate retrieval index
        retrieval_delay = delay_steps
        
        # Slice history
        # We need a sequence of length RNN_SEQUENCE_LENGTH ending at -(delay + 1)
        most_recent_idx = -(retrieval_delay + 1)
        oldest_idx = most_recent_idx - cfg.RNN_SEQUENCE_LENGTH + 1
        
        # Extract sequence manually from deque to match agent logic
        buffer_list = list(leader_q_hist) # Snapshot
        buffer_qd_list = list(leader_qd_hist)
        
        # Safety check
        if len(buffer_list) < cfg.RNN_SEQUENCE_LENGTH + retrieval_delay:
            continue
            
        hist_q = np.array(buffer_list[oldest_idx : most_recent_idx+1 if most_recent_idx != -1 else None])
        hist_qd = np.array(buffer_qd_list[oldest_idx : most_recent_idx+1 if most_recent_idx != -1 else None])
        
        # Normalize scalar
        norm_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        
        # Tensor
        delay_col = np.full((len(hist_q), 1), norm_delay, dtype=np.float32)
        input_np = np.hstack([hist_q, hist_qd, delay_col])
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device).float()
        
        # 2. Autoregressive Inference
        steps_to_predict = int(norm_delay * cfg.DELAY_INPUT_NORM_FACTOR)
        
        if steps_to_predict <= 0:
            pred_q = hist_q[-1]
        else:
            with torch.no_grad():
                _, hidden = model.lstm(input_tensor)
                
                last_obs = input_tensor[0, -1, :]
                curr_q = last_obs[:cfg.N_JOINTS].clone()
                curr_qd = last_obs[cfg.N_JOINTS:2*cfg.N_JOINTS].clone()
                curr_delay_scalar = norm_delay
                
                for _ in range(steps_to_predict):
                    delay_t = torch.tensor([curr_delay_scalar], device=device)
                    inp = torch.cat([curr_q, curr_qd, delay_t], dim=0).view(1, 1, -1)
                    
                    residual, hidden = model.forward_step(inp, hidden)
                    
                    delta = residual[0] * cfg.TARGET_DELTA_SCALE
                    delta = torch.clamp(delta, -0.2, 0.2)
                    
                    curr_q = curr_q + delta[:cfg.N_JOINTS]
                    curr_qd = curr_qd + delta[cfg.N_JOINTS:]
                    curr_delay_scalar += dt_norm
            
            pred_q = curr_q.cpu().numpy()
            
        # --- C. Recording ---
        history_truth_q.append(q_real)
        history_pred_q.append(pred_q)
        history_error.append(np.linalg.norm(q_real - pred_q))
        history_delay.append(delay_steps)

    # 4. Visualization
    history_truth_q = np.array(history_truth_q)
    history_pred_q = np.array(history_pred_q)
    history_error = np.array(history_error)
    time_axis = np.arange(len(history_error)) * dt
    
    print(f"\nResults:")
    print(f"Avg Error: {np.mean(history_error):.4f} rad")
    print(f"Max Error: {np.max(history_error):.4f} rad")
    
    # Plot 7 Joints
    fig, axes = plt.subplots(8, 1, figsize=(12, 20), sharex=True)
    
    for i in range(7):
        ax = axes[i]
        ax.plot(time_axis, history_truth_q[:, i], 'k-', label='Truth', linewidth=1.5)
        ax.plot(time_axis, history_pred_q[:, i], 'r--', label='Pred', linewidth=1.0)
        ax.set_ylabel(f"J{i+1}")
        ax.grid(True)
        if i == 0: ax.legend(loc='upper right')
        
    # Error Plot
    axes[7].plot(time_axis, history_error, 'b-', label='L2 Error')
    axes[7].set_ylabel("Error (rad)")
    axes[7].set_xlabel("Time (s)")
    axes[7].grid(True)
    axes[7].legend()
    
    plt.tight_layout()
    plt.savefig("lstm_7joint_validation.png")
    print("Plot saved to lstm_7joint_validation.png")
    plt.show()

if __name__ == "__main__":
    main()