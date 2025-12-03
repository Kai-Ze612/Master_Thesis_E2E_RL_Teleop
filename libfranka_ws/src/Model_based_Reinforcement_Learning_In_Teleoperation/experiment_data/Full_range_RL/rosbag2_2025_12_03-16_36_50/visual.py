import argparse
import sqlite3
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import PointStamped

import rosbag2_py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_bag_data(bag_path, local_topic, remote_topic):
    """
    Reads a ROS2 bag and extracts PointStamped messages into Pandas DataFrames.
    """
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        return None, None

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    local_data = []
    remote_data = []

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        if topic in [local_topic, remote_topic]:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            
            # Extract data (timestamp in nanoseconds)
            row = {
                'timestamp': t,
                'x': msg.point.x,
                'y': msg.point.y,
                'z': msg.point.z
            }
            
            if topic == local_topic:
                local_data.append(row)
            else:
                remote_data.append(row)

    # Convert to DataFrames
    df_local = pd.DataFrame(local_data)
    df_remote = pd.DataFrame(remote_data)
    
    return df_local, df_remote

def align_and_calculate_error(df_local, df_remote):
    """
    1. Trims first/last 2 seconds.
    2. Synchronizes streams.
    3. Calculates error.
    """
    if df_local.empty or df_remote.empty:
        print("Error: One or both topics contain no data.")
        return None

    # 1. Sort by time
    df_local = df_local.sort_values('timestamp')
    df_remote = df_remote.sort_values('timestamp')

    # 2. Trim Data (Remove first 2s and last 2s)
    # Find global start and end to determine the window
    t_min = min(df_local['timestamp'].iloc[0], df_remote['timestamp'].iloc[0])
    t_max = max(df_local['timestamp'].iloc[-1], df_remote['timestamp'].iloc[-1])
    
    trim_nanos = 2 * 1_000_000_000 # 2 seconds in nanoseconds
    
    start_cutoff = t_min + trim_nanos
    end_cutoff = t_max - trim_nanos
    
    print(f"Trimming data...")
    print(f"  Original Range: {t_min} to {t_max}")
    print(f"  Trimmed Range:  {start_cutoff} to {end_cutoff}")

    df_local = df_local[(df_local['timestamp'] >= start_cutoff) & (df_local['timestamp'] <= end_cutoff)]
    df_remote = df_remote[(df_remote['timestamp'] >= start_cutoff) & (df_remote['timestamp'] <= end_cutoff)]

    if df_local.empty or df_remote.empty:
        print("Error: Data is empty after trimming first/last 2 seconds.")
        return None

    # 3. Time Alignment (merge_asof)
    # [FIX] tolerance must be int because timestamps are int64
    tolerance_ns = int(50_000_000) # 50ms
    
    df_merged = pd.merge_asof(
        df_local, 
        df_remote, 
        on='timestamp', 
        suffixes=('_local', '_remote'), 
        direction='nearest',
        tolerance=tolerance_ns 
    )
    
    # Drop rows where alignment failed (too large gap)
    original_len = len(df_merged)
    df_merged = df_merged.dropna()
    print(f"Alignment complete. Matched {len(df_merged)}/{original_len} samples.")

    # 4. Normalize time to start at 0
    start_time = df_merged['timestamp'].iloc[0]
    df_merged['time_sec'] = (df_merged['timestamp'] - start_time) / 1e9

    # 5. Calculate Euclidean Error
    df_merged['error_x'] = df_merged['x_local'] - df_merged['x_remote']
    df_merged['error_y'] = df_merged['y_local'] - df_merged['y_remote']
    df_merged['error_z'] = df_merged['z_local'] - df_merged['z_remote']
    
    df_merged['euclidean_error'] = np.sqrt(
        df_merged['error_x']**2 + 
        df_merged['error_y']**2 + 
        df_merged['error_z']**2
    )

    return df_merged

def visualize_results(df):
    if df is None: return

    # --- [FIX] ROBUST STYLE SELECTION ---
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # Fallback for older matplotlib versions
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Total Tracking Error
    # [CRITICAL FIX: CONVERT TO NUMPY ARRAY]
    axes[0].plot(df['time_sec'].to_numpy(), df['euclidean_error'].to_numpy(), color='#d62728', linewidth=1.5)
    axes[0].set_title("Total Tracking Error (Euclidean Distance)", fontsize=14)
    axes[0].set_ylabel("Error (m)", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 2. Component Trajectories (X, Y, Z) - All plotting arguments now use .to_numpy()
    axes[1].plot(df['time_sec'].to_numpy(), df['x_local'].to_numpy(), label='Local X', linestyle='-', color='tab:blue')
    axes[1].plot(df['time_sec'].to_numpy(), df['x_remote'].to_numpy(), label='Remote X', linestyle='--', color='tab:blue')
    axes[1].plot(df['time_sec'].to_numpy(), df['y_local'].to_numpy(), label='Local Y', linestyle='-', color='tab:green')
    axes[1].plot(df['time_sec'].to_numpy(), df['y_remote'].to_numpy(), label='Remote Y', linestyle='--', color='tab:green')
    axes[1].plot(df['time_sec'].to_numpy(), df['z_local'].to_numpy(), label='Local Z', linestyle='-', color='tab:orange')
    axes[1].plot(df['time_sec'].to_numpy(), df['z_remote'].to_numpy(), label='Remote Z', linestyle='--', color='tab:orange')
    axes[1].set_title("End-Effector Trajectories (Components)", fontsize=14)
    axes[1].set_ylabel("Position (m)", fontsize=12)
    axes[1].legend(loc='upper right', ncol=3, fontsize=8)

    # 3. Component Errors - All plotting arguments now use .to_numpy()
    axes[2].plot(df['time_sec'].to_numpy(), df['error_x'].to_numpy(), label='Error X', color='tab:blue', alpha=0.8)
    axes[2].plot(df['time_sec'].to_numpy(), df['error_y'].to_numpy(), label='Error Y', color='tab:green', alpha=0.8)
    axes[2].plot(df['time_sec'].to_numpy(), df['error_z'].to_numpy(), label='Error Z', color='tab:orange', alpha=0.8)
    axes[2].axhline(0, color='black', linewidth=0.8)
    axes[2].set_title("Component Errors", fontsize=14)
    axes[2].set_ylabel("Deviation (m)", fontsize=12)
    axes[2].set_xlabel("Time (seconds)", fontsize=12)
    axes[2].legend()

    # Calculate Statistics
    rmse = np.sqrt((df['euclidean_error']**2).mean())
    mean_err = df['euclidean_error'].mean()
    max_err = df['euclidean_error'].max()

    stats_text = (
        f"RMSE: {rmse:.4f} m\n"
        f"Mean Error: {mean_err:.4f} m\n"
        f"Max Error: {max_err:.4f} m"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(0.02, 0.95, stats_text, transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('tracking_performance_analysis.png', dpi=300)
    print(f"Analysis saved to tracking_performance_analysis.png")
    print(f"Statistics:\n{stats_text}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze ROS2 Bag for Tracking Error")
    parser.add_argument('bag_path', type=str, help="Path to the ROS2 bag folder")
    args = parser.parse_args()

    local_topic = '/local_robot/ee_pose'
    remote_topic = '/remote_robot/ee_pose'

    print(f"Reading bag: {args.bag_path}...")
    df_local, df_remote = read_bag_data(args.bag_path, local_topic, remote_topic)
    
    print(f"Local samples: {len(df_local) if df_local is not None else 0}")
    print(f"Remote samples: {len(df_remote) if df_remote is not None else 0}")

    if df_local is not None and not df_local.empty:
        df_aligned = align_and_calculate_error(df_local, df_remote)
        visualize_results(df_aligned)

if __name__ == "__main__":
    main()