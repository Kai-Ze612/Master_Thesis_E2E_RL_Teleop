import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys

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
# Visualization Speed (1.0 = Real Time, 0.5 = Slow Motion)
PLAYBACK_SPEED = 1.0 

def main():
    print("--- MUJOCO TRAJECTORY VISUALIZER ---")
    print(f"Model Path: {cfg.DEFAULT_MUJOCO_MODEL_PATH}")
    print(f"Control Freq: {cfg.DEFAULT_CONTROL_FREQ} Hz")
    
    # 1. Initialize Generator (IK Logic)
    sim_generator = LocalRobotSimulator(
        trajectory_type=TrajectoryType.FIGURE_8,
        randomize_params=False 
    )
    
    # 2. Initialize MuJoCo for Visualization
    # We load the model independently to attach the viewer
    try:
        m = mujoco.MjModel.from_xml_path(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        return

    # Reset Generator
    q_start, _ = sim_generator.reset()
    
    # Set initial state in Viewer
    d.qpos[:cfg.N_JOINTS] = q_start
    mujoco.mj_forward(m, d)

    print("\nLaunching Viewer... (Close window to stop)")
    
    # 3. Visualization Loop
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        # Camera Setup (Optional: Adjust view)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.5, 0.0, 0.5]
        
        start_time = time.time()
        sim_time = 0.0
        dt = 1.0 / cfg.DEFAULT_CONTROL_FREQ
        
        while viewer.is_running():
            step_start = time.time()

            # --- A. Step the Generator (IK) ---
            # This calculates the next q using your IK logic
            q_target, qd_target, _, _, _, _ = sim_generator.step()
            
            # --- B. Update MuJoCo Viewer State ---
            # We strictly set qpos to visualize the kinematic result
            d.qpos[:cfg.N_JOINTS] = q_target
            d.qvel[:cfg.N_JOINTS] = qd_target
            
            # Forward kinematics to update visuals (geoms/sites)
            mujoco.mj_forward(m, d)
            
            # Update the viewer
            viewer.sync()
            
            # --- C. Timing Control ---
            # Ensure we watch it at realistic speed
            time_spent = time.time() - step_start
            sleep_time = (dt / PLAYBACK_SPEED) - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            sim_time += dt
            
            # Optional: Print progress every second
            if int(sim_time / dt) % cfg.DEFAULT_CONTROL_FREQ == 0:
                print(f"Sim Time: {sim_time:.1f}s | J4 Pos: {q_target[3]:.3f}")

if __name__ == "__main__":
    main()