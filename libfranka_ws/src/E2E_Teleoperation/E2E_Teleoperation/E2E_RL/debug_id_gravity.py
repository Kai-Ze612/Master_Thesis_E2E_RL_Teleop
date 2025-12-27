import mujoco
import numpy as np
import time
import os
from pathlib import Path

# Configuration
MODEL_PATH = "franka_description/mujoco/franka/panda.xml" # UPDATE THIS PATH if needed
# If you are running from E2E_RL folder, you might need to adjust the path or use the one from config
import E2E_Teleoperation.config.robot_config as cfg

def test_gravity_compensation():
    print("========================================")
    print("   MUJOCO INVERSE DYNAMICS TEST")
    print("========================================")

    # 1. Load Model
    model_path = str(cfg.DEFAULT_MUJOCO_MODEL_PATH)
    if not os.path.exists(model_path):
        # Fallback to local if config path is wrong
        model_path = "panda.xml" 
    
    print(f"Loading Model: {model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # 2. Check Gravity
    print(f"Model Gravity: {model.opt.gravity}")
    if np.linalg.norm(model.opt.gravity) < 0.1:
        print("[WARNING] Gravity is nearly ZERO! ID will fail to find holding torque.")
    
    # 3. Setup a challenging pose (Arm extended)
    # Joint order: 1, 2, 3, 4, 5, 6, 7
    # Extend elbow (-1.5) and shoulder to create moment arm
    q_hold = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.785])
    
    data.qpos[:7] = q_hold
    data.qvel[:7] = 0.0
    data.qacc[:7] = 0.0 # We want ZERO acceleration (Hold)
    
    # Forward kinematics to update body positions
    mujoco.mj_forward(model, data)
    
    print(f"Initial Joint Positions: {data.qpos[:7]}")
    
    # 4. Run Inverse Dynamics
    # This calculates: tau = M*qacc + C*qvel + g(q)
    # Since qvel=0, qacc=0, this is purely g(q) (Gravity Compensation)
    mujoco.mj_inverse(model, data)
    
    id_torque = data.qfrc_inverse[:7].copy()
    print(f"\nCalculated Holding Torque (Gravity Comp): \n{id_torque}")
    
    # 5. Apply Torque and Step Simulation (Forward Dynamics)
    print("\nSimulating 1000 steps (Hold Test)...")
    
    # Reset data to apply control
    data.qpos[:7] = q_hold
    data.qvel[:7] = 0.0
    mujoco.mj_forward(model, data)
    
    error_history = []
    
    for i in range(1000):
        # Apply the Calculated ID Torque
        # Note: We assume Actuators are direct Torque Motors (gear=1)
        data.ctrl[:7] = id_torque
        
        # Step Physics
        mujoco.mj_step(model, data)
        
        # Check drift
        curr_q = data.qpos[:7]
        err = np.linalg.norm(curr_q - q_hold)
        error_history.append(err)
        
        # Visual debug (optional, prints every 100 steps)
        if i % 100 == 0:
            print(f"Step {i}: Drift Error = {err:.5f} rad")

    total_drift = error_history[-1]
    print(f"\nFinal Drift after 1s: {total_drift:.5f} rad")
    
    if total_drift < 0.01:
        print("\n[SUCCESS] Gravity Compensation Works! ID is correct.")
    else:
        print("\n[FAILURE] Robot Drifted. ID Torque did not hold the robot.")
        print("Possible causes:")
        print("1. 'gravcomp' is still active in XML (set to 0 or remove).")
        print("2. Actuator 'gear' is not 1.0.")
        print("3. Damping/Friction in XML is high, but mj_inverse expects it.")

if __name__ == "__main__":
    test_gravity_compensation()