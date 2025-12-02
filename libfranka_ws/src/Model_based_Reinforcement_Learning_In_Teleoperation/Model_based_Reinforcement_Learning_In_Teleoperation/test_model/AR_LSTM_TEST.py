"""
Validation Script for Step-Based Autoregressive LSTM.
Simulates real-time deployment where the LSTM must bridge a variable delay.

CORRECTIONS:
1. Implements Normalization (Q_MEAN, Q_STD) to match Training Env.
2. Implements EMA Filtering to match Training Env.
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
TEST_DURATION_SEC = 20.0  
MODEL_PATH = cfg.LSTM_MODEL_PATH

# --- HELPER FUNCTIONS (Match training_env.py logic) ---
def normalize_state(q, qd):
    q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
    qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
    return np.concatenate([q_norm, qd_norm])

def normalize_input(q, qd, delay_scalar):
    state_norm = normalize_state(q, qd)
    return np.concatenate([state_norm, [delay_scalar]])

def denormalize_state(pred_norm):
    q_norm = pred_norm[:7]
    qd_norm = pred_norm[7:]
    q = (q_norm * cfg.Q_STD) + cfg.Q_MEAN
    qd = (qd_norm * cfg.QD_STD) + cfg.QD_MEAN
    return np.concatenate([q, qd])
# ------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STEP-BASED LSTM VALIDATION ---")
    print(f"Device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    # 1. Load Model
    model = StateEstimator(
        input_dim_total=int(cfg.ESTIMATOR_STATE_DIM),
        output_dim=int(cfg.N_JOINTS * 2)
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'state_estimator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_estimator_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Setup Simulation
    dt = 1.0 / cfg.DEFAULT_CONTROL_FREQ
    total_steps = int(TEST_DURATION_SEC / dt)
    
    local_sim = LocalRobotSimulator(
        trajectory_type=TrajectoryType.LISSAJOUS_COMPLEX, 
        randomize_params=False
    )
    local_sim.reset()
    delay_sim = DelaySimulator(cfg.DEFAULT_CONTROL_FREQ, TEST_CONFIG, seed=101)
    
    # 3. Buffers
    history_buffer_q = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
    history_buffer_qd = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
    
    # Pre-fill
    q_start, qd_start, _, _, _, _ = local_sim.step()
    for _ in range(cfg.RNN_SEQUENCE_LENGTH + 50):
        history_buffer_q.append(q_start)
        history_buffer_qd.append(qd_start)

    # Metrics
    log_truth = []
    log_pred = []
    log_delay_steps = []
    
    # EMA State
    prediction_ema = None
    ema_alpha = cfg.PREDICTION_EMA_ALPHA

    print(f"Simulating {total_steps} steps...")
    
    for t in tqdm(range(total_steps)):
        # --- A. Physics Step ---
        q_real, qd_real, _, _, _, _ = local_sim.step()
        
        # --- B. Delay Simulation ---
        current_history_len = len(history_buffer_q)
        delay_steps = delay_sim.get_observation_delay_steps(current_history_len)
        
        history_buffer_q.append(q_real)
        history_buffer_qd.append(qd_real)
        
        # --- C. Agent Inference ---
        
        # 1. Prepare NORMALIZED Input Sequence
        delayed_idx = -(1 + delay_steps)
        full_q = np.array(history_buffer_q)
        full_qd = np.array(history_buffer_qd)
        
        end_idx = len(full_q) + delayed_idx + 1
        start_idx = end_idx - int(cfg.RNN_SEQUENCE_LENGTH)
        
        if start_idx < 0: continue
        
        # Get raw slice
        seq_q = full_q[start_idx:end_idx]
        seq_qd = full_qd[start_idx:end_idx]
        
        # Normalize Sequence
        norm_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        seq_norm_buffer = []
        for i in range(len(seq_q)):
            step_norm = normalize_input(seq_q[i], seq_qd[i], norm_delay)
            seq_norm_buffer.append(step_norm)
            
        input_tensor = torch.tensor(np.array(seq_norm_buffer), dtype=torch.float32).unsqueeze(0).to(device)
        
        # 2. Autoregressive Loop
        pred_q_now = None
        
        with torch.no_grad():
            # A. Process History
            _, hidden = model.lstm(input_tensor)
            
            # B. Prepare First Input (Last frame of context)
            curr_input = input_tensor[:, -1:, :] 
            
            # Calculate steps to bridge
            steps_to_predict = min(delay_steps, int(cfg.MAX_AR_STEPS))
            
            if delay_steps == 0:
                # No prediction needed, use ground truth
                pred_q_now = seq_q[-1]
            else:
                curr_state_norm = None
                dt_norm_step = (1.0 / cfg.DEFAULT_CONTROL_FREQ) / cfg.DELAY_INPUT_NORM_FACTOR
                
                # Loop
                for _ in range(steps_to_predict):
                    # Predict next state (Normalized)
                    pred_state_norm, hidden = model.forward_step(curr_input, hidden)
                    curr_state_norm = pred_state_norm
                    
                    # Update Delay (Decrement)
                    curr_delay_val = curr_input[0, 0, -1].item()
                    next_delay_val = max(0.0, curr_delay_val - dt_norm_step)
                    
                    # Prepare next input
                    delay_t = torch.tensor([[[next_delay_val]]], device=device)
                    curr_input = torch.cat([pred_state_norm, delay_t], dim=2)
                
                # Denormalize final result
                final_pred_norm = curr_state_norm.cpu().numpy()[0, 0]
                final_pred_denorm = denormalize_state(final_pred_norm)
                
                # Extract Q (first 7 dims)
                pred_q_now = final_pred_denorm[:cfg.N_JOINTS]

        # --- D. Apply EMA Filter ---
        if prediction_ema is None:
            prediction_ema = pred_q_now
        else:
            prediction_ema = ema_alpha * pred_q_now + (1.0 - ema_alpha) * prediction_ema
            
        final_output_q = prediction_ema

        # --- E. Logging ---
        log_truth.append(q_real)
        log_pred.append(final_output_q)
        log_delay_steps.append(delay_steps)

    # 4. Analysis & Plotting
    log_truth = np.array(log_truth)
    log_pred = np.array(log_pred)
    log_delay_steps = np.array(log_delay_steps)
    
    min_len = min(len(log_truth), len(log_pred))
    log_truth = log_truth[:min_len]
    log_pred = log_pred[:min_len]
    log_delay_steps = log_delay_steps[:min_len]
    
    errors = np.linalg.norm(log_truth - log_pred, axis=1)
    
    print("\n--- Validation Results ---")
    print(f"Mean Prediction Error: {np.mean(errors):.5f} rad")
    print(f"Max Prediction Error:  {np.max(errors):.5f} rad")
    
    # Plotting
    time_axis = np.arange(len(log_truth)) * dt
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Joint 1
    axs[0].plot(time_axis, log_truth[:, 0], 'k-', label='Ground Truth')
    axs[0].plot(time_axis, log_pred[:, 0], 'r--', label='LSTM Prediction')
    axs[0].set_ylabel('Joint 1 (rad)')
    axs[0].legend()
    axs[0].set_title(f"Trajectory Tracking ({TEST_CONFIG.name})")
    
    # Joint 4
    axs[1].plot(time_axis, log_truth[:, 3], 'k-')
    axs[1].plot(time_axis, log_pred[:, 3], 'r--')
    axs[1].set_ylabel('Joint 4 (rad)')
    
    # Error
    axs[2].plot(time_axis, errors, 'b-')
    axs[2].set_ylabel('L2 Error (rad)')
    axs[2].grid(True)
    
    # Delay
    axs[3].plot(time_axis, log_delay_steps, 'g-', alpha=0.6)
    axs[3].set_ylabel('Delay (steps)')
    axs[3].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig("lstm_validation_normalized.png")
    print(f"\nPlot saved to: lstm_validation_normalized.png")
    plt.show()

if __name__ == "__main__":
    main()