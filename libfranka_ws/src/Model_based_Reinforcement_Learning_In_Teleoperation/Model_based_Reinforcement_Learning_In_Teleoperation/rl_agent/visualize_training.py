import time
import numpy as np
import mujoco
import mujoco.viewer
import sys
import os

# Ensure the project root is in the python path
sys.path.append(os.getcwd())

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import N_JOINTS, DEFAULT_CONTROL_FREQ

def visualize():
    # 1. Initialize the Environment
    # We use 'render_mode=None' because we handle the viewer manually here.
    # You can change trajectory_type to 'SQUARE' or 'FIGURE_8' to see different paths.
    env = TeleoperationEnvWithDelay(
        delay_config=ExperimentConfig.LOW_DELAY,
        trajectory_type=TrajectoryType.FIGURE_8,
        render_mode=None 
    )

    print("="*60)
    print("Initializing MuJoCo Passive Viewer...")
    print(f"Robot Model: {env.remote_robot.model_path}")
    print("="*60)

    # 2. Launch the Passive Viewer
    # This attaches a window to the existing physics model of the remote robot
    with mujoco.viewer.launch_passive(env.remote_robot.model, env.remote_robot.data) as viewer:
        
        # Initial Reset
        obs, info = env.reset()
        
        # Toggle options (optional, wireframe etc can be set in viewer GUI)
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        
        print("Running Simulation... Press Ctrl+C in terminal to stop.")
        
        step_count = 0
        
        while viewer.is_running():
            step_count += 1
            
            # 3. Define Action
            # We send a 7-dimensional ZERO vector.
            # In your training_env.py, sending 7 dims (instead of 21) triggers 
            # the "Data Collection Mode" logic where the robot follows the 
            # delayed target using only the baseline PD controller.
            # This matches your log: "Predicted q: None (Data Collection Mode)"
            action = np.zeros(N_JOINTS) 
            
            # 4. Step the Environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 5. Sync the Viewer
            # This pushes the updated physics state (qpos, qvel) to the window
            viewer.sync()
            
            # 6. Sleep to maintain Real-Time speed
            # Without this, the simulation runs as fast as the CPU allows (too fast to watch)
            time.sleep(1.0 / DEFAULT_CONTROL_FREQ)
            
            # Optional: Print info every 100 steps to verify tracking
            if step_count % 100 == 0:
                print(f"Step: {step_count} | Error: {info.get('real_time_joint_error', 0):.4f} | Delay: {info.get('current_delay_steps', 0)}")

            # Handle Episode End
            if terminated or truncated:
                print(f"Episode finished at step {step_count}. Resetting...")
                env.reset()
                step_count = 0

if __name__ == "__main__":
    try:
        visualize()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")