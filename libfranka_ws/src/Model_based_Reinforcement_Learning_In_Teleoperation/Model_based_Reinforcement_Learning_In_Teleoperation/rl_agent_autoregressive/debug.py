import time
import numpy as np
import torch
import sys
import os

# Adjust path so imports work
sys.path.append(os.getcwd())

# Import your modules
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import TrajectoryType
from sbsp_wrapper import SBSP_Trajectory_Wrapper

def debug_main():
    print("1. [START] Starting Debug Script...")
    
    # 1. Config
    config = ExperimentConfig.LOW_DELAY
    traj = TrajectoryType.FIGURE_8
    
    print("2. [INIT] Creating TeleoperationEnvWithDelay...")
    try:
        # Base Env
        env = TeleoperationEnvWithDelay(
            delay_config=config,
            trajectory_type=traj,
            randomize_trajectory=False,
            render_mode=None, # No rendering to isolate physics issues
            lstm_model_path=None 
        )
        print("   [SUCCESS] Base Env created.")
    except Exception as e:
        print(f"   [FAIL] Base Env creation failed: {e}")
        return

    print("3. [WRAP] Applying SBSP Wrapper...")
    try:
        # Wrapper
        env = SBSP_Trajectory_Wrapper(env, n_models=2) # Reduced models to 2 for speed test
        print("   [SUCCESS] SBSP Wrapper applied.")
        print(f"   [CHECK] Device being used: {env.dc_models[0].device}")
    except Exception as e:
        print(f"   [FAIL] Wrapper application failed: {e}")
        return

    print("4. [RESET] Resetting Environment (This creates the Mujoco Physics)...")
    try:
        obs, info = env.reset()
        print("   [SUCCESS] Reset complete.")
        print(f"   [INFO] Obs Shape: {obs.shape}")
    except Exception as e:
        print(f"   [FAIL] Reset failed (Physics crash?): {e}")
        return

    print("5. [LOOP] Attempting 2000 Simulation Steps...")
    start_time = time.time()
    
    for i in range(2000):
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 100 == 0:
            print(f"   -> Step {i}/2000 completed. (Prediction Error: {info.get('prediction_error', -1):.4f})")
            
        # FORCE TRAINING (The part that might be slow)
        # We manually force the buffer full to trigger the SBSP learning
        if i == 500:
            print("   [TRIGGER] Filling replay buffer to force SBSP Learning...")
            while len(env.replay_buffer) < 1001:
                 # Fake data
                env.replay_buffer.append((np.zeros(21), np.zeros(14)))
            print("   [TRIGGER] Buffer filled. Next step will trigger Neural Network Training.")

    end_time = time.time()
    duration = end_time - start_time
    print(f"6. [DONE] Finished 2000 steps in {duration:.2f} seconds.")
    print(f"   [SPEED] {2000/duration:.2f} Steps/Sec")

if __name__ == "__main__":
    debug_main()