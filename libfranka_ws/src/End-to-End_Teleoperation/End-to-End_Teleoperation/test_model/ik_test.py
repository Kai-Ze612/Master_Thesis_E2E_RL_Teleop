import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# --- IMPORTS ---
# Ensure python path sees the workspace root
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import (
    LocalRobotSimulator, TrajectoryType
)
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

# --- CONFIG ---
DURATION_SEC = 60.0
FREQ = cfg.DEFAULT_CONTROL_FREQ  # e.g., 200Hz
TOTAL_STEPS = int(DURATION_SEC * FREQ)

def main():
    print(f"--- IK SOLVER DEBUG TOOL ---")
    print(f"Simulating {DURATION_SEC}s at {FREQ}Hz ({TOTAL_STEPS} steps)")
    
    # 1. Initialize Simulator
    # This uses the EXACT class from your training code
    sim = LocalRobotSimulator(
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_params=False  # Deterministic for debugging
    )
    
    # 2. Reset
    q_start, _ = sim.reset()
    print(f"Start Config: {np.round(q_start, 3)}")
    
    # 3. Data Collection
    history_q = []
    history_qd = []
    
    # Pre-fill with start config for continuity in plots
    history_q.append(q_start)
    history_qd.append(np.zeros_like(q_start))
    
    print("Generating trajectory...")
    for _ in tqdm(range(TOTAL_STEPS)):
        # Step the simulator (Calls IK Internally)
        q, qd, _, _, _, _ = sim.step()
        
        history_q.append(q.copy())
        history_qd.append(qd.copy())
        
    history_q = np.array(history_q)
    history_qd = np.array(history_qd)
    
    # 4. Plotting
    time_axis = np.linspace(0, DURATION_SEC, len(history_q))
    
    # Figure 1: Joint Positions
    fig1, axes1 = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    fig1.suptitle("IK Solver Output: Joint Positions (q)", fontsize=16)
    
    for i in range(7):
        ax = axes1[i]
        ax.plot(time_axis, history_q[:, i], 'b-', linewidth=1.5)
        ax.set_ylabel(f"J{i+1} (rad)")
        ax.grid(True)
        
        # Check for Jump Discontinuities
        diffs = np.abs(np.diff(history_q[:, i]))
        max_jump = np.max(diffs)
        if max_jump > 0.1: # 0.1 rad jump in 0.005s is HUGE
            ax.set_title(f"Joint {i+1} - [WARNING] Max Jump: {max_jump:.3f} rad", color='red')
        else:
            ax.set_title(f"Joint {i+1}")

    axes1[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Figure 2: Joint Velocities
    fig2, axes2 = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    fig2.suptitle("IK Solver Output: Joint Velocities (qd)", fontsize=16)
    
    limit = 1.5 # Your safety clip limit
    
    for i in range(7):
        ax = axes2[i]
        ax.plot(time_axis, history_qd[:, i], 'r-', linewidth=1.0)
        
        # Draw limits
        ax.axhline(limit, color='k', linestyle='--', alpha=0.5)
        ax.axhline(-limit, color='k', linestyle='--', alpha=0.5)
        
        ax.set_ylabel(f"J{i+1} (rad/s)")
        ax.grid(True)
        
        # Check for saturation
        max_vel = np.max(np.abs(history_qd[:, i]))
        if max_vel >= limit - 0.01:
             ax.set_title(f"Joint {i+1} - [SATURATED] Peak: {max_vel:.2f} rad/s", color='orange')
        else:
             ax.set_title(f"Joint {i+1} - Peak: {max_vel:.2f} rad/s")

    axes2[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    fig1.savefig("ik_positions.png")
    fig2.savefig("ik_velocities.png")
    print("\nPlots saved to 'ik_positions.png' and 'ik_velocities.png'")
    plt.show()

if __name__ == "__main__":
    main()