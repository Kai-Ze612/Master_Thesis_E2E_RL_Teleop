# test_local_robot_simulator_figure8.py
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
sys.path.append(os.getcwd())

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import (
    LocalRobotSimulator,
    TrajectoryType
)
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import DEFAULT_CONTROL_FREQ

print("FINAL TEST: LocalRobotSimulator generating Figure-8")
print("Watch the robot — this applies the logic from your working IK script.\n")

# 1. Initialize Simulator
leader = LocalRobotSimulator(
    trajectory_type=TrajectoryType.FIGURE_8,
    randomize_params=False
)

# 2. CRITICAL: Reset BEFORE getting the data handle
# This ensures we get the active MjData object, not an old one.
leader.reset()

# 3. Get references (Just like in your working script)
model = leader.model
data = leader.data 

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    start_time = time.time()

    while viewer.is_running():
        step += 1
        
        # 4. Get the next joint configuration from the simulator
        q, qd, _, _, _, _ = leader.step()
        
        # ---------------------------------------------------------
        # THE FIX (Borrowed from test_ik_solver_trajectory.py)
        # ---------------------------------------------------------
        
        # A. Explicitly write the new q to the viewer's data handle
        # (This ensures the viewer sees exactly what the solver produced)
        data.qpos[:len(q)] = q
        data.qvel[:len(qd)] = qd
        
        # B. Force Forward Kinematics
        # This calculates the 3D position of the robot meshes based on qpos
        # Without this, the robot stays frozen even if numbers change.
        mujoco.mj_forward(model, data)

        # ---------------------------------------------------------

        if step % 50 == 0:
            j6 = q[5]
            print(f"Step {step:4d} | Joint6: {j6:+.6f} rad ({np.degrees(j6):+7.1f}°)")
        
        # 5. Sync the viewer
        viewer.sync()
        
        # Timing control (keep it real-time)
        time_until_next_step = (start_time + (step / DEFAULT_CONTROL_FREQ)) - time.time()
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("\nTest finished.")