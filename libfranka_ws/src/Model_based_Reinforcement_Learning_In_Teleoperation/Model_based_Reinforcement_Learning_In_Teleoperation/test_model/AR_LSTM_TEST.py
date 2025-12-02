"""
Validation Script for Step-Based Autoregressive LSTM.
Simulates real-time deployment where the LSTM must bridge a variable delay
using the 'Constant Delay' semantics (Delay = Age of Observation).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import deque

# --- IMPORTS ---
# Ensure we can import from the package
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
TEST_DURATION_SEC = 20.0  # Seconds to simulate

# [FIX] Use the Absolute Path provided
MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent_autoregressive/lstm_training_output/LSTM_AR_FULL_RANGE_COVER_20251202_131939/best_model.pth"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STEP-BASED LSTM VALIDATION ---")
    print(f"Device: {device}")
    print(f"Delay Config: {TEST_CONFIG.name}")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please check the absolute path.")
        return

    print(f"Loading model from: {MODEL_PATH}")
    
    # Initialize model (15D Input -> 14D Output)
    model = StateEstimator(
        input_dim_total=int(cfg.ESTIMATOR_STATE_DIM),
        output_dim=int(cfg.N_JOINTS * 2)
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'state_estimator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_estimator_state_dict'])
    elif 'actor_state_dict' not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        print("Error: Checkpoint format unknown or is an SAC policy.")
        return
        
    model.eval()

    # 2. Setup Simulation Components
    dt = 1.0 / cfg.DEFAULT_CONTROL_FREQ
    total_steps = int(TEST_DURATION_SEC / dt)
    
    # Test on LISSAJOUS to check generalization
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
    
    print(f"Simulating {total_steps} steps...")
    
    for t in tqdm(range(total_steps)):
        # --- A. Physics Step ---
        q_real, qd_real, _, _, _, _ = local_sim.step()
        
        # --- B. Delay Simulation ---
        current_history_len = len(history_buffer_q)
        delay_steps = delay_sim.get_observation_delay_steps(current_history_len)
        
        # Update history (Ground Truth)
        history_buffer_q.append(q_real)
        history_buffer_qd.append(qd_real)
        
        # --- C. Agent Inference ---
        
        # 1. Prepare Input Sequence
        # We need the sequence ending 'delay_steps' ago
        delayed_idx = -(1 + delay_steps)
        
        # Slice indices
        full_q = np.array(history_buffer_q)
        full_qd = np.array(history_buffer_qd)
        
        end_idx = len(full_q) + delayed_idx + 1
        start_idx = end_idx - int(cfg.RNN_SEQUENCE_LENGTH)
        
        if start_idx < 0: continue
        
        seq_q = full_q[start_idx:end_idx]
        seq_qd = full_qd[start_idx:end_idx]
        
        # 2. Add Constant Delay Feature
        norm_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        delay_col = np.full((len(seq_q), 1), norm_delay, dtype=np.float32)
        
        input_np = np.hstack([seq_q, seq_qd, delay_col]) # (Seq, 15)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).float().to(device)
        
        # 3. Autoregressive Loop
        pred_q_now = None
        
        with torch.no_grad():
            # A. Process History
            _, hidden = model.lstm(input_tensor)
            
            # B. Prepare First Input for Loop
            # Last step of the sequence (t - delay)
            curr_input = input_tensor[:, -1:, :] # (1, 1, 15)
            
            # Extract constant delay feature
            curr_delay = curr_input[:, :, -1:]
            
            # Steps to bridge
            # If delay is 0, we already have the state
            if delay_steps == 0:
                pred_q_now = seq_q[-1]
            else:
                steps_to_predict = min(delay_steps, int(cfg.MAX_AR_STEPS))
                curr_state = None
                
                for _ in range(steps_to_predict):
                    # Predict next state (14D)
                    pred_state, hidden = model.forward_step(curr_input, hidden)
                    curr_state = pred_state
                    
                    # Prepare next input:
                    # [State (14D) + Constant Delay (1D)] -> 15D
                    # Delay remains CONSTANT (Option A logic)
                    curr_input = torch.cat([pred_state, curr_delay], dim=2)
                
                # Extract Q (first 7 dims)
                pred_q_now = curr_state.cpu().numpy()[0, 0, :cfg.N_JOINTS]

        # --- D. Logging ---
        log_truth.append(q_real)
        log_pred.append(pred_q_now)
        log_delay_steps.append(delay_steps)

    # 4. Analysis
    log_truth = np.array(log_truth)
    log_pred = np.array(log_pred)
    
    # Calculate errors
    min_len = min(len(log_truth), len(log_pred))
    log_truth = log_truth[:min_len]
    log_pred = log_pred[:min_len]
    log_delay_steps = log_delay_steps[:min_len]
    
    errors = np.linalg.norm(log_truth - log_pred, axis=1)
    
    print("\n--- Validation Results ---")
    print(f"Mean Prediction Error: {np.mean(errors):.5f} rad")
    print(f"Max Prediction Error:  {np.max(errors):.5f} rad")
    print(f"Avg Delay Bridged:     {np.mean(log_delay_steps):.1f} steps")

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
    output_plot = "lstm_step_validation_results.png"
    plt.savefig(output_plot)
    print(f"\nPlot saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()