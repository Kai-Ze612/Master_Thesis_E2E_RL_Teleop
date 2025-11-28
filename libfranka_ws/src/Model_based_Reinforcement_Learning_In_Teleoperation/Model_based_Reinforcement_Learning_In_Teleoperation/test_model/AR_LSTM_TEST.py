import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from tqdm import tqdm

# --- IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import (
    Figure8TrajectoryGenerator, 
    TrajectoryParams
)
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

# --- TEST PARAMETERS ---
TEST_CONFIG = ExperimentConfig.FULL_RANGE_COVER 
JOINT_IDX_TO_PLOT = 3
TEST_DURATION_SEC = 60.0
PLOT_SEGMENT_SEC = 60.0


def generate_figure8_data(duration_sec, dt):
    """Generates ground truth (q, qd) for the full duration."""
    params = TrajectoryParams() 
    generator = Figure8TrajectoryGenerator(params)
    
    steps = int(duration_sec / dt)
    t = np.linspace(0, duration_sec, steps)
    freq = cfg.TRAJECTORY_FREQUENCY
    
    q_full = np.zeros((steps, cfg.N_JOINTS))
    qd_full = np.zeros((steps, cfg.N_JOINTS))
    
    for j in range(cfg.N_JOINTS):
        amp = 0.2 + (0.05 * j)  
        phase = 0.5 * j
        q_full[:, j] = cfg.INITIAL_JOINT_CONFIG[j] + amp * np.sin(2 * np.pi * freq * t + phase)
        qd_full[:, j] = (amp * 2 * np.pi * freq) * np.cos(2 * np.pi * freq * t + phase)
        
    return q_full.astype(np.float32), qd_full.astype(np.float32)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- LSTM FULL TRAJECTORY VALIDATION (FIXED) ---")
    print(f"Config: DELTA_SCALE={cfg.TARGET_DELTA_SCALE}, NORM_FACTOR={cfg.DELAY_INPUT_NORM_FACTOR}")
    print(f"Test Duration: {TEST_DURATION_SEC}s ({int(TEST_DURATION_SEC * cfg.DEFAULT_CONTROL_FREQ)} steps)")

    # 1. Load Model
    model_path = cfg.LSTM_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = "checkpoints/lstm/best_model.pth" 
    
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
    print("Model loaded successfully.")

    # 2. Setup Data & Delay
    dt = 1.0 / cfg.DEFAULT_CONTROL_FREQ
    total_steps = int(TEST_DURATION_SEC / dt)
    q_gt, qd_gt = generate_figure8_data(TEST_DURATION_SEC, dt)
    
    delay_sim = DelaySimulator(cfg.DEFAULT_CONTROL_FREQ, TEST_CONFIG, seed=42)
    
    # 3. Simulation Loop
    real_time_errors = []
    real_time_errors_old = []  # For comparison
    predicted_traj = []
    predicted_traj_old = []
    ground_truth_traj = []
    delays_encountered = []
    
    seq_len = cfg.RNN_SEQUENCE_LENGTH
    start_idx = seq_len + 100 
    
    print(f"Running simulation with BOTH old and fixed inference...")
    dt_norm = dt / cfg.DELAY_INPUT_NORM_FACTOR
    
    try:
        for t in tqdm(range(start_idx, total_steps)):
            # A. Determine Delay
            delay_steps = delay_sim.get_observation_delay_steps(seq_len)
            delays_encountered.append(delay_steps)
            
            delayed_idx = t - delay_steps
            
            if delayed_idx < seq_len:
                continue

            # B. Construct Input Sequence
            hist_q = q_gt[delayed_idx - seq_len : delayed_idx]
            hist_qd = qd_gt[delayed_idx - seq_len : delayed_idx]
            
            norm_delay = float(delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
            
            delay_col = np.full((seq_len, 1), norm_delay, dtype=np.float32)
            input_np = np.hstack([hist_q, hist_qd, delay_col])
            input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device)
            
            steps_to_predict = delay_steps
            
            # ============================================================
            # OLD METHOD (your original - for comparison)
            # ============================================================
            if steps_to_predict <= 0:
                pred_q_old = hist_q[-1]
            else:
                with torch.no_grad():
                    _, hidden_old = model.lstm(input_tensor)
                    
                    last_obs = input_tensor[0, -1, :]
                    curr_q = last_obs[:cfg.N_JOINTS].clone()
                    curr_qd = last_obs[cfg.N_JOINTS:2*cfg.N_JOINTS].clone()
                    curr_delay_scalar = norm_delay
                    
                    for _ in range(steps_to_predict):
                        delay_t = torch.tensor([curr_delay_scalar], device=device)
                        inp = torch.cat([curr_q, curr_qd, delay_t], dim=0).view(1, 1, -1)
                        
                        residual, hidden_old = model.forward_step(inp, hidden_old)
                        
                        delta = residual[0] * cfg.TARGET_DELTA_SCALE
                        delta = torch.clamp(delta, -0.2, 0.2)
                        
                        curr_q = curr_q + delta[:cfg.N_JOINTS]
                        curr_qd = curr_qd + delta[cfg.N_JOINTS:]
                        curr_delay_scalar += dt_norm
                
                pred_q_old = curr_q.cpu().numpy()
            
            # ============================================================
            # FIXED METHOD (matching training exactly)
            # ============================================================
            if steps_to_predict <= 0:
                pred_q_fixed = hist_q[-1]
            else:
                with torch.no_grad():
                    # KEY FIX: Truncate sequence like training does!
                    # Training: cutoff_idx = RNN_SEQUENCE_LENGTH - packet_loss_steps
                    cutoff_idx = seq_len - steps_to_predict
                    cutoff_idx = max(1, cutoff_idx)  # At least 1 step
                    
                    safe_history = input_tensor[:, :cutoff_idx, :]
                    
                    # Build hidden state from TRUNCATED sequence
                    _, hidden_fixed = model.lstm(safe_history)
                    
                    # Start from END of truncated sequence (not full!)
                    current_input = safe_history[:, -1:, :].clone()  # Shape: (1, 1, 15)
                    
                    # AR Loop - matching training structure exactly
                    for _ in range(steps_to_predict):
                        pred_delta, hidden_fixed = model.forward_step(current_input, hidden_fixed)
                        
                        # Update state using tensor operations (like training)
                        last_known_state = current_input[:, :, :14]
                        predicted_next_state = last_known_state + (pred_delta.unsqueeze(1) * cfg.TARGET_DELTA_SCALE)
                        
                        # Optional clamp
                        # predicted_next_state = torch.clamp(predicted_next_state, -10, 10)
                        
                        current_delay_norm = current_input[:, :, 14:15]
                        next_delay_norm = current_delay_norm + dt_norm
                        
                        current_input = torch.cat([predicted_next_state, next_delay_norm], dim=2)
                    
                    pred_q_fixed = current_input[0, 0, :cfg.N_JOINTS].cpu().numpy()

            # D. Record Metrics
            true_q = q_gt[t]
            
            error_old = np.linalg.norm(true_q - pred_q_old)
            error_fixed = np.linalg.norm(true_q - pred_q_fixed)
            
            real_time_errors_old.append(error_old)
            real_time_errors.append(error_fixed)
            predicted_traj_old.append(pred_q_old)
            predicted_traj.append(pred_q_fixed)
            ground_truth_traj.append(true_q)

    except KeyboardInterrupt:
        print("Simulation stopped manually.")
    
    # 4. Analysis & Plotting
    real_time_errors = np.array(real_time_errors)
    real_time_errors_old = np.array(real_time_errors_old)
    predicted_traj = np.array(predicted_traj)
    predicted_traj_old = np.array(predicted_traj_old)
    ground_truth_traj = np.array(ground_truth_traj)
    delays_encountered = np.array(delays_encountered)
    
    avg_rmse_fixed = np.mean(real_time_errors)
    avg_rmse_old = np.mean(real_time_errors_old)
    max_error_fixed = np.max(real_time_errors)
    max_error_old = np.max(real_time_errors_old)
    avg_delay = np.mean(delays_encountered)
    
    print(f"\n--- RESULTS COMPARISON ---")
    print(f"OLD METHOD:")
    print(f"  Average RMSE: {avg_rmse_old:.6f} rad")
    print(f"  Maximum Error: {max_error_old:.6f} rad")
    print(f"\nFIXED METHOD:")
    print(f"  Average RMSE: {avg_rmse_fixed:.6f} rad")
    print(f"  Maximum Error: {max_error_fixed:.6f} rad")
    print(f"\nImprovement: {(avg_rmse_old - avg_rmse_fixed) / avg_rmse_old * 100:.1f}%")
    print(f"Average Delay: {avg_delay:.1f} steps")
    
    # Plotting - 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    time_axis = np.arange(len(real_time_errors)) * dt
    j = JOINT_IDX_TO_PLOT
    
    # Subplot 1: Trajectory comparison
    ax1.plot(time_axis, ground_truth_traj[:, j], 'k-', label='Ground Truth', linewidth=2)
    ax1.plot(time_axis, predicted_traj_old[:, j], 'r--', label='OLD (Wrong)', linewidth=1, alpha=0.7)
    ax1.plot(time_axis, predicted_traj[:, j], 'g-', label='FIXED', linewidth=1.5)
    ax1.set_title(f"Trajectory Tracking (Joint {j+1}) - Full Duration ({TEST_DURATION_SEC}s)")
    ax1.set_ylabel("Position (rad)")
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: Error comparison
    ax2.plot(time_axis, real_time_errors_old, 'r-', label=f'OLD (avg={avg_rmse_old:.4f})', linewidth=1, alpha=0.7)
    ax2.plot(time_axis, real_time_errors, 'g-', label=f'FIXED (avg={avg_rmse_fixed:.4f})', linewidth=1)
    ax2.set_title("Prediction Error Comparison (L2 Norm)")
    ax2.set_ylabel("Error (rad)")
    ax2.legend()
    ax2.grid(True)
    
    # Subplot 3: Delay over time
    ax3.plot(np.arange(len(delays_encountered)) * dt, delays_encountered, 'b-', linewidth=0.5)
    ax3.set_title(f"Delay Steps (avg={avg_delay:.1f})")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Delay (steps)")
    ax3.grid(True)
    
    plt.tight_layout()
    save_path = "lstm_validation_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plots saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()