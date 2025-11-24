"""
Script to read ROS2 bag and compute/plot EE positions using MuJoCo FK.
Correctly uses the Stores Enum for rosbags 0.11.0+.
Prints total duration to verify data length.
"""
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- Import the Enum and Factory function ---
try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import get_typestore, Stores
except ImportError as e:
    print(f"Error importing rosbags: {e}")
    print("Please install: pip install rosbags pandas matplotlib")
    sys.exit(1)
# -------------------------------------------------

# Import your config
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    DEFAULT_MUJOCO_MODEL_PATH, 
    EE_BODY_NAME, 
    N_JOINTS
)

# --- CONFIGURATION ---
# Absolute path to your bag data folder
BAG_PATH = '/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/Visualization/28/11/my_experiment_dat_2' 

LOCAL_TOPIC = '/local_robot/joint_states'
REMOTE_TOPIC = '/franka/joint_states'
# ---------------------

def get_ee_position(model, data, q_pos):
    """Compute Forward Kinematics using MuJoCo."""
    # Set joint positions
    data.qpos[:N_JOINTS] = q_pos[:N_JOINTS]
    # Forward kinematics
    mujoco.mj_kinematics(model, data)
    # Get EE position
    return data.body(EE_BODY_NAME).xpos.copy()

def get_robust_typestore():
    """Factory to get the correct Typestore object from the Enum."""
    selected_store = None
    
    if hasattr(Stores, 'ROS2_HUMBLE'):
        print("Using typestore config: ROS2_HUMBLE")
        selected_store = Stores.ROS2_HUMBLE
    elif hasattr(Stores, 'LATEST'):
        print("Using typestore config: LATEST")
        selected_store = Stores.LATEST
    elif hasattr(Stores, 'ROS2_FOXY'):
        print("Using typestore config: ROS2_FOXY")
        selected_store = Stores.ROS2_FOXY
    else:
        print("Warning: Specific ROS2 store not found. Using first available.")
        selected_store = list(Stores)[0]

    return get_typestore(selected_store)

def process_bag():
    # Load MuJoCo Model for FK
    print(f"Loading model from: {DEFAULT_MUJOCO_MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(DEFAULT_MUJOCO_MODEL_PATH)
    data = mujoco.MjData(model)

    local_times = []
    local_ee_pos = []
    
    remote_times = []
    remote_ee_pos = []

    bag_path = Path(BAG_PATH)
    if not bag_path.exists():
        print(f"Error: Bag file not found at {bag_path}")
        return

    # Initialize Type Store
    typestore = get_robust_typestore()

    print("Reading bag file...")
    
    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic in [LOCAL_TOPIC, REMOTE_TOPIC]]
            
            if not connections:
                print(f"No connections found! Checked for topics: {LOCAL_TOPIC}, {REMOTE_TOPIC}")
                print(f"Available topics in bag: {[c.topic for c in reader.connections]}")
                return

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                q = np.array(msg.position)
                
                # Compute FK
                xyz = get_ee_position(model, data, q)
                
                if connection.topic == LOCAL_TOPIC:
                    local_times.append(timestamp)
                    local_ee_pos.append(xyz)
                elif connection.topic == REMOTE_TOPIC:
                    remote_times.append(timestamp)
                    remote_ee_pos.append(xyz)
    except Exception as e:
        print(f"Error reading bag: {e}")
        import traceback
        traceback.print_exc()
        return

    # Convert to numpy
    local_ee_pos = np.array(local_ee_pos)
    remote_ee_pos = np.array(remote_ee_pos)
    
    # Calculate Data Statistics
    if len(local_times) > 0 and len(remote_times) > 0:
        start_time = min(local_times[0], remote_times[0])
        end_time = max(local_times[-1], remote_times[-1])
        total_duration_sec = (end_time - start_time) * 1e-9
        
        local_t = (np.array(local_times) - start_time) * 1e-9
        remote_t = (np.array(remote_times) - start_time) * 1e-9
        
        # --- PRINT SUMMARY ---
        print("\n" + "="*40)
        print(" DATA SUMMARY")
        print("="*40)
        print(f"Local Samples:   {len(local_t)}")
        print(f"Remote Samples:  {len(remote_t)}")
        print(f"Total Duration:  {total_duration_sec:.2f} seconds")
        print("="*40 + "\n")
        # ---------------------

        # --- PLOTTING ---
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        # X Axis
        ax[0].plot(local_t, local_ee_pos[:, 0], label='Local (Leader)', color='blue')
        ax[0].plot(remote_t, remote_ee_pos[:, 0], label='Remote (Follower)', color='orange', linestyle='--')
        ax[0].set_ylabel('X Position (m)')
        ax[0].set_title(f'Trajectory Tracking (Duration: {total_duration_sec:.1f}s)')
        ax[0].legend()
        ax[0].grid(True)

        # Y Axis
        ax[1].plot(local_t, local_ee_pos[:, 1], color='blue')
        ax[1].plot(remote_t, remote_ee_pos[:, 1], color='orange', linestyle='--')
        ax[1].set_ylabel('Y Position (m)')
        ax[1].grid(True)

        # Z Axis
        ax[2].plot(local_t, local_ee_pos[:, 2], color='blue')
        ax[2].plot(remote_t, remote_ee_pos[:, 2], color='orange', linestyle='--')
        ax[2].set_ylabel('Z Position (m)')
        ax[2].set_xlabel('Time (s)')
        ax[2].grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print("No data found in bag for specified topics!")

if __name__ == "__main__":
    process_bag()