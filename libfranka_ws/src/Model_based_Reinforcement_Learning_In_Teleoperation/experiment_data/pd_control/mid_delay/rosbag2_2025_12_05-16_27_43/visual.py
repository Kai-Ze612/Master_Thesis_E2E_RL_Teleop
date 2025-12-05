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
    
    MODIFICATION: The storage_id is set to '', allowing rosbag2 to automatically 
    select the correct storage plugin (MCAP or SQLite3) based on the metadata.yaml.
    """
    reader = rosbag2_py.SequentialReader()
    
    # --- CRITICAL MODIFICATION FOR MCAP/SQLITE COMPATIBILITY ---
    # Set storage_id to an empty string. rosbag2 will read metadata.yaml 
    # and automatically initialize the correct storage plugin (e.g., 'mcap').
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='')
    # -----------------------------------------------------------
    
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        print("NOTE: The error suggests that the required storage plugin (likely 'mcap')")
        print("failed to initialize. Please ensure the 'ros-*-rosbag2-storage-mcap' package")
        print("is installed for your ROS distribution.")
        return None, None

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    local_data = []
    remote_data = []

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        if topic in [local_topic, remote_topic]:
            # Guard against topics existing in metadata but having no corresponding type
            if topic not in type_map:
                print(f"Warning: Topic '{topic}' found but message type is unknown.")
                continue
                
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

def align_and_calculate_error_2d(df_local, df_remote):
    """
    1. Trims first/last 2 seconds.
    2. Synchronizes streams (nearest alignment).
    3. Calculates 2D Euclidean Error (X-Y plane only).
    """
    if df_local.empty or df_remote.empty:
        print("Error: One or both topics contain no data.")
        return None

    # 1. Sort by time
    df_local = df_local.sort_values('timestamp')
    df_remote = df_remote.sort_values('timestamp')

    # 2. Trim Data (Remove first 2s and last 2s)
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
    tolerance_ns = int(50_000_000) # 50ms tolerance
    
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

    # 5. Calculate 2D Euclidean Error (X-Y Plane Only)
    df_merged['error_x'] = df_merged['x_local'] - df_merged['x_remote']
    df_merged['error_y'] = df_merged['y_local'] - df_merged['y_remote']
    
    df_merged['euclidean_error'] = np.sqrt(
        df_merged['error_x']**2 + 
        df_merged['error_y']**2
    )

    return df_merged

def visualize_results_2d(df):
    """
    Generates 3 plots: Total Error, Component Trajectories, and X-Y Path Comparison.
    """
    if df is None: return

    # --- STYLE SELECTION ---
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')

    # Use 3 subplots: 1. Error (2D), 2. Components (X, Y), 3. 2D Path Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # 1. Total Tracking Error (2D)
    axes[0].plot(df['time_sec'].to_numpy(), df['euclidean_error'].to_numpy(), color='#d62728', linewidth=1.5)
    axes[0].set_title("Total Tracking Error (Euclidean Distance in X-Y Plane)", fontsize=14)
    axes[0].set_ylabel("Error (m)", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 2. Component Trajectories (X, Y only)
    axes[1].plot(df['time_sec'].to_numpy(), df['x_local'].to_numpy(), label='Local X (Reference)', linestyle='-', color='tab:blue')
    axes[1].plot(df['time_sec'].to_numpy(), df['x_remote'].to_numpy(), label='Remote X (Actual)', linestyle='--', color='tab:blue')
    axes[1].plot(df['time_sec'].to_numpy(), df['y_local'].to_numpy(), label='Local Y (Reference)', linestyle='-', color='tab:green')
    axes[1].plot(df['time_sec'].to_numpy(), df['y_remote'].to_numpy(), label='Remote Y (Actual)', linestyle='--', color='tab:green')
    axes[1].set_title("End-Effector Trajectories (X and Y Components)", fontsize=14)
    axes[1].set_ylabel("Position (m)", fontsize=12)
    axes[1].legend(loc='upper right', ncol=2, fontsize=10)

    # 3. 2D Path Visualization (New Plot)
    axes[2].plot(df['x_local'].to_numpy(), df['y_local'].to_numpy(), label='Local Path (Reference)', linestyle='-', color='#1f77b4', linewidth=2)
    axes[2].plot(df['x_remote'].to_numpy(), df['y_remote'].to_numpy(), label='Remote Path (Actual)', linestyle='--', color='#ff7f0e', linewidth=2)
    
    # Add start/end markers
    axes[2].plot(df['x_local'].iloc[0], df['y_local'].iloc[0], 'o', color='#1f77b4', markersize=8, label='Start')
    axes[2].plot(df['x_remote'].iloc[-1], df['y_remote'].iloc[-1], 'x', color='#ff7f0e', markersize=8, label='End')
    
    axes[2].set_title("X-Y Plane Trajectory Comparison", fontsize=14)
    axes[2].set_xlabel("X Position (m)", fontsize=12)
    axes[2].set_ylabel("Y Position (m)", fontsize=12)
    axes[2].axis('equal') # Ensure correct aspect ratio for spatial analysis
    axes[2].legend(loc='upper right', fontsize=10)

    # Calculate Statistics (Based on 2D Error)
    rmse = np.sqrt((df['euclidean_error']**2).mean())
    mean_err = df['euclidean_error'].mean()
    max_err = df['euclidean_error'].max()

    stats_text = (
        f"2D RMSE: {rmse:.4f} m\n"
        f"2D Mean Error: {mean_err:.4f} m\n"
        f"2D Max Error: {max_err:.4f} m"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(0.02, 0.95, stats_text, transform=axes[0].transAxes, fontsize=10,
                  verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('tracking_performance_analysis_2d.png', dpi=300)
    print(f"Analysis saved to tracking_performance_analysis_2d.png")
    print(f"Statistics:\n{stats_text}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze ROS2 Bag for Tracking Error in the X-Y Plane")
    parser.add_argument('bag_path', type=str, help="Path to the ROS2 bag folder")
    args = parser.parse_args()

    local_topic = '/local_robot/ee_pose'
    remote_topic = '/remote_robot/ee_pose'

    print(f"Reading bag: {args.bag_path}...")
    df_local, df_remote = read_bag_data(args.bag_path, local_topic, remote_topic)
    
    print(f"Local samples: {len(df_local) if df_local is not None else 0}")
    print(f"Remote samples: {len(df_remote) if df_remote is not None else 0}")

    if df_local is not None and not df_local.empty and df_remote is not None and not df_remote.empty:
        df_aligned = align_and_calculate_error_2d(df_local, df_remote) 
        visualize_results_2d(df_aligned)
    elif df_local is None or df_remote is None:
        print("Analysis aborted due to error opening bag or missing data.")
    else:
        print("Analysis aborted: One or both dataframes are empty before alignment.")

if __name__ == "__main__":
    main()