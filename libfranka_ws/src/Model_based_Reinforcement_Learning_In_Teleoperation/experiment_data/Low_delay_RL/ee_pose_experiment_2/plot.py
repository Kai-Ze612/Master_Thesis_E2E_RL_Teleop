"""
Script to plot 2D End-Effector tracking (X, Y) from ROS2 bag.
Calculates error strictly in the XY plane (ignoring Z).
Units: Meters (m)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- Robust Import for rosbags ---
try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import get_typestore, Stores
except ImportError as e:
    print(f"Error importing rosbags: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
# 1. Path to your recorded bag folder
BAG_PATH = '/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/experiment_data/Low_delay_RL/ee_pose_experiment_2'

# 2. Topic Names
LOCAL_TOPIC = '/local_robot/ee_pose'
REMOTE_TOPIC = '/remote_robot/ee_pose'

# 3. Analysis Settings
TRIM_START = 2.0  # Skip first 2.0 seconds
TRIM_END = 2.0    # Skip last 2.0 seconds
# ---------------------

def get_robust_typestore():
    """Factory to get the correct Typestore object."""
    if hasattr(Stores, 'ROS2_HUMBLE'):
        return get_typestore(Stores.ROS2_HUMBLE)
    elif hasattr(Stores, 'LATEST'):
        return get_typestore(Stores.LATEST)
    else:
        return get_typestore(list(Stores)[0])

def extract_data(bag_path):
    local_data = {'t': [], 'x': [], 'y': []}
    remote_data = {'t': [], 'x': [], 'y': []}
    
    path = Path(bag_path)
    if not path.exists():
        print(f"Error: Bag not found at {path}")
        return None, None

    typestore = get_robust_typestore()
    
    print(f"Reading bag: {path.name}...")
    
    with AnyReader([path], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in [LOCAL_TOPIC, REMOTE_TOPIC]]
        
        if not connections:
            print("No topics found!")
            return None, None

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            # Calculate time in seconds
            t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # ONLY extracting X and Y. Z is ignored.
            if connection.topic == LOCAL_TOPIC:
                local_data['t'].append(t_sec)
                local_data['x'].append(msg.point.x)
                local_data['y'].append(msg.point.y)
            else:
                remote_data['t'].append(t_sec)
                remote_data['x'].append(msg.point.x)
                remote_data['y'].append(msg.point.y)

    # Convert to numpy arrays
    for data in [local_data, remote_data]:
        if len(data['t']) > 0:
            data['t'] = np.array(data['t'])
            data['x'] = np.array(data['x'])
            data['y'] = np.array(data['y'])
    
    # Normalize time to start at 0.0
    if len(local_data['t']) > 0 and len(remote_data['t']) > 0:
        start_time = min(local_data['t'][0], remote_data['t'][0])
        local_data['t'] -= start_time
        remote_data['t'] -= start_time
        
    return local_data, remote_data

def plot_results(local, remote):
    if len(local['t']) == 0 or len(remote['t']) == 0:
        print("Empty data sequences.")
        return

    # 1. Interpolate Local data to match Remote timestamps
    local_x_interp = np.interp(remote['t'], local['t'], local['x'])
    local_y_interp = np.interp(remote['t'], local['t'], local['y'])
    
    # 2. Calculate 2D Error (Meters)
    error_x = local_x_interp - remote['x']
    error_y = local_y_interp - remote['y']
    error_2d = np.sqrt(error_x**2 + error_y**2)

    # 3. Calculate Statistics with Double Trimming
    total_duration = remote['t'][-1]
    start_cutoff = TRIM_START
    end_cutoff = total_duration - TRIM_END
    
    # Create mask: (Time > 2s) AND (Time < Total - 2s)
    mask = (remote['t'] > start_cutoff) & (remote['t'] < end_cutoff)
    
    if np.any(mask):
        avg_error = np.mean(error_2d[mask])
        max_error = np.max(error_2d[mask])
        rmse_error = np.sqrt(np.mean(error_2d[mask]**2))
        analyzed_duration = end_cutoff - start_cutoff
    else:
        print(f"Warning: Trimming removed all data! Duration: {total_duration:.2f}s")
        avg_error = 0.0
        max_error = 0.0
        rmse_error = 0.0
        analyzed_duration = 0.0

    print("="*40)
    print(f"TRACKING STATISTICS (Meters)")
    print(f"Window: {start_cutoff:.1f}s to {end_cutoff:.1f}s (Duration: {analyzed_duration:.1f}s)")
    print("="*40)
    print(f"Avg 2D Error: {avg_error:.5f} m")
    print(f"Max 2D Error: {max_error:.5f} m")
    print(f"RMSE 2D:      {rmse_error:.5f} m")
    print("="*40)

    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Trajectory Tracking", fontsize=14)

    # Top Left: X vs Time
    axs[0, 0].plot(local['t'], local['x'], label='Leader (Local)', color='blue', linewidth=2)
    axs[0, 0].plot(remote['t'], remote['x'], label='Follower (Remote)', color='orange', linestyle='--')
    axs[0, 0].set_title("X Position vs Time")
    axs[0, 0].set_ylabel("X (m)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Top Right: Y vs Time
    axs[0, 1].plot(local['t'], local['y'], label='Leader', color='blue', linewidth=2)
    axs[0, 1].plot(remote['t'], remote['y'], label='Follower', color='orange', linestyle='--')
    axs[0, 1].set_title("Y Position vs Time")
    axs[0, 1].set_ylabel("Y (m)")
    axs[0, 1].grid(True)

    # Bottom Left: Trajectory Shape (Y vs X)
    axs[1, 0].plot(local['x'], local['y'], label='Reference Path', color='blue', alpha=0.6)
    axs[1, 0].plot(remote['x'], remote['y'], label='Actual Path', color='orange', linestyle='--', alpha=0.8)
    axs[1, 0].set_title("2D Path (Top-Down View)")
    axs[1, 0].set_xlabel("X (m)")
    axs[1, 0].set_ylabel("Y (m)")
    axs[1, 0].axis('equal') 
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Bottom Right: Error vs Time (in Meters)
    axs[1, 1].plot(remote['t'], error_2d, color='red', label='Error')
    
    # Draw lines for cutoffs
    axs[1, 1].axvline(x=start_cutoff, color='black', linestyle=':', label='Start Cutoff')
    axs[1, 1].axvline(x=end_cutoff, color='black', linestyle='-.', label='End Cutoff')
    
    axs[1, 1].set_title("2D Euclidean Error (m)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Error (m)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    # Add text box with stats in Meters
    stats_text = f"Window: {start_cutoff}-{end_cutoff}s\nAvg: {avg_error:.4f} m\nMax: {max_error:.4f} m"
    axs[1, 1].text(0.05, 0.80, stats_text, transform=axs[1, 1].transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    l_data, r_data = extract_data(BAG_PATH)
    if l_data and r_data:
        plot_results(l_data, r_data)