"""
Validation Script for 10-Step Chunk Autoregressive LSTM.
Simulates real-time deployment where the LSTM must bridge a variable delay
by predicting consecutive 10-step 'shots'.
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
SHOT_SIZE = 10            # Matches your training configuration

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 10-STEP LSTM VALIDATION ---")
    print(f"Device: {device}")
    print(f"Delay Config: {TEST_CONFIG.name}")

    # 1. Load Model
    # We use the final_model.pth or best_model.pth from the output dir
    # Update this path to where your model actually is
    model_path = os.path.join(cfg.CHECKPOINT_DIR_LSTM, "final_model.pth")
    # Use fallback if specific file not found, for safety in this script
    if not os.path.exists(model_path):
        model_path = cfg.LSTM_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please check the path or run training first.")
        return

    print(f"Loading model from: {model_path}")
    
    # Initialize model with correct dimensions (15D input -> 14D*10 output)
    model = StateEstimator(
        input_dim_total=cfg.ESTIMATOR_STATE_DIM,
        output_dim=cfg.N_JOINTS * 2,
        shot_size=SHOT_SIZE
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_estimator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_estimator_state_dict'])
    elif 'actor_state_dict' not in checkpoint: # Avoid loading SAC checkpoint by mistake
        model.load_state_dict(checkpoint)
    else:
        print("Error: Checkpoint seems to be an SAC policy, not an LSTM model.")
        return
        
    model.eval()

    # 2. Setup Simulation Components
    dt = 1.0 / cfg.DEFAULT_CONTROL_FREQ
    total_steps = int(TEST_DURATION_SEC / dt)
    
    # We use Lissajous to test generalization (since you likely trained on Figure-8)
    local_sim = LocalRobotSimulator(
        trajectory_type=TrajectoryType.LISSAJOUS_COMPLEX, 
        randomize_params=False
    )
    local_sim.reset()
    
    delay_sim = DelaySimulator(cfg.DEFAULT_CONTROL_FREQ, TEST_CONFIG, seed=101)
    
    # 3. Buffers (Simulating the Agent's memory)
    # Stores [q, qd] pairs
    history_buffer = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
    
    # Pre-fill buffer to avoid cold-start issues
    q_start, qd_start, _, _, _, _ = local_sim.step()
    init_state = np.concatenate([q_start, qd_start])
    for _ in range(cfg.RNN_SEQUENCE_LENGTH + 50):
        history_buffer.append(init_state)

    # Metrics Storage
    log_truth = []
    log_pred = []
    log_delay_steps = []
    
    print(f"Simulating {total_steps} steps...")
    
    for t in tqdm(range(total_steps)):
        # --- A. Physics Step ---
        q_real, qd_real, _, _, _, _ = local_sim.step()
        real_state = np.concatenate([q_real, qd_real])
        
        # --- B. Delay Simulation ---
        # 1. Calculate how old the data available to the agent is
        current_history_len = len(history_buffer)
        delay_steps = delay_sim.get_observation_delay_steps(current_history_len)
        
        # 2. Update "Ground Truth History" with the new real state
        # In a real network, this append happens 'delay_steps' later. 
        # Here, we append immediately but 'look back' to simulate reading old data.
        history_buffer.append(real_state)
        
        # --- C. Agent Inference (Bridging the Gap) ---
        
        # 1. Construct Input Sequence (The "Past")
        # We need the sequence ending at (now - delay_steps)
        # deque index -1 is 'now'. index -(1 + delay) is the delayed head.
        delayed_head_idx = -(1 + delay_steps)
        
        # Slice the buffer
        # We need RNN_SEQUENCE_LENGTH elements ending at delayed_head_idx
        # Converting deque to list is slow in loop, but acceptable for validation script
        full_hist = np.array(history_buffer)
        
        # Indices for slicing
        end_idx = len(full_hist) + delayed_head_idx + 1 # +1 for exclusive upper bound
        start_idx = end_idx - cfg.RNN_SEQUENCE_LENGTH
        
        if start_idx < 0: continue # Buffer safety
        
        seq_data = full_hist[start_idx:end_idx] # Shape (150, 14)
        
        # Add Normalized Delay Feature (15th dimension)
        norm_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        delay_col = np.full((len(seq_data), 1), norm_delay, dtype=np.float32)
        input_np = np.hstack([seq_data, delay_col]) # Shape (150, 15)
        
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).float().to(device)
        
        # 2. Autoregressive Prediction Loop (The "Future")
        # We need to bridge 'delay_steps'.
        # The model predicts chunks of 'SHOT_SIZE' (10).
        
        pred_q_now = None
        
        with torch.no_grad():
            # Initial LSTM pass over history
            _, hidden = model.lstm(input_tensor)
            
            # Prepare for AR loop
            curr_input = input_tensor[:, -1:, :] # The last delayed observation (1, 1, 15)
            steps_covered = 0
            
            # Prediction trajectory container
            predicted_trajectory = []
            
            # We assume the model predicts Absolute States [q, qd] directly 
            # (based on your get_future_target_chunk logic)
            
            while steps_covered < delay_steps:
                # Predict 10 steps ahead
                shot_pred, hidden = model.forward_shot(curr_input, hidden)
                # shot_pred shape: (1, 10, 14)
                
                # Store prediction
                shot_np = shot_pred.cpu().numpy()[0] # (10, 14)
                predicted_trajectory.append(shot_np)
                
                # Prepare input for next shot
                # We take the LAST step of the predicted shot
                last_pred_state = shot_pred[:, -1:, :] # (1, 1, 14)
                
                # Update delay feature for the next input
                # Time has advanced by SHOT_SIZE steps
                # Note: Depending on training, delay input might need to decrease (converging to 0) 
                # or strictly represent the "distance from reality".
                # Usually in AR, we keep the delay feature or increment it? 
                # In your `training_env.py`: `curr_delay_scalar += dt_norm_chunk`
                norm_delay += (SHOT_SIZE * dt) / cfg.DELAY_INPUT_NORM_FACTOR
                delay_t = torch.tensor([[[norm_delay]]], device=device)
                
                curr_input = torch.cat([last_pred_state, delay_t], dim=2)
                
                steps_covered += SHOT_SIZE
            
            # 3. Extract the specific point corresponding to "Now"
            # We generated chunks of 10. Total length = N * 10.
            # We need the state at index `delay_steps`.
            
            flat_traj = np.concatenate(predicted_trajectory, axis=0) # (N*10, 14)
            
            # Clamp index to bounds
            target_idx = min(delay_steps, len(flat_traj) - 1)
            pred_state_now = flat_traj[target_idx]
            pred_q_now = pred_state_now[:cfg.N_JOINTS]

        # --- D. Logging ---
        log_truth.append(q_real)
        log_pred.append(pred_q_now)
        log_delay_steps.append(delay_steps)

    # 4. Analysis & Plotting
    log_truth = np.array(log_truth)
    log_pred = np.array(log_pred)
    errors = np.linalg.norm(log_truth - log_pred, axis=1)
    
    print("\n--- Validation Results ---")
    print(f"Mean Prediction Error: {np.mean(errors):.5f} rad")
    print(f"Max Prediction Error:  {np.max(errors):.5f} rad")
    print(f"Avg Delay Bridged:     {np.mean(log_delay_steps):.1f} steps")

    # Plotting
    time_axis = np.arange(len(log_truth)) * dt
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Plot Joint 1
    axs[0].plot(time_axis, log_truth[:, 0], 'k-', label='Ground Truth')
    axs[0].plot(time_axis, log_pred[:, 0], 'r--', label='LSTM Prediction')
    axs[0].set_ylabel('Joint 1 Position (rad)')
    axs[0].legend()
    axs[0].set_title(f"Trajectory Tracking ({TEST_CONFIG.name})")
    
    # Plot Joint 4 (Usually highly active)
    axs[1].plot(time_axis, log_truth[:, 3], 'k-')
    axs[1].plot(time_axis, log_pred[:, 3], 'r--')
    axs[1].set_ylabel('Joint 4 Position (rad)')
    
    # Plot Error
    axs[2].plot(time_axis, errors, 'b-')
    axs[2].set_ylabel('L2 Error (rad)')
    axs[2].set_ylim(0, max(0.2, np.max(errors)*1.1))
    
    # Plot Delay
    axs[3].plot(time_axis, log_delay_steps, 'g-', alpha=0.6)
    axs[3].set_ylabel('Delay (steps)')
    axs[3].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig("lstm_10step_validation_results.png")
    print("\nPlot saved to: lstm_10step_validation_results.png")
    plt.show()

if __name__ == "__main__":
    main()