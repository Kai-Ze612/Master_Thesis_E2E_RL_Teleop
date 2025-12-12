#!/usr/bin/env python3
"""
FIXED: Convert rosbag2 teleoperation data to a single combined CSV file.
- Handles geometry_msgs/msg/Point (which has no header).
- Uses bag timestamps.
- Corrected topic names.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    sys.exit(1)

# ---------------- CONFIGURATION ---------------- #
TARGET_TOPICS = {
    'leader_pose': '/leader/ee_position',   # CHANGED from ee_pose
    'remote_pose': '/remote/ee_position'    # CHANGED from ee_pose
}
MAX_DURATION_SEC = 50
# ----------------------------------------------- #

def extract_xyz(msg):
    """
    Helper to extract x,y,z from different message types 
    (Point, Pose, PoseStamped)
    """
    # 1. Try accessing as Point (msg.x) - This matches your current data
    if hasattr(msg, 'x') and hasattr(msg, 'y'):
        return msg.x, msg.y, msg.z
    
    # 2. Try accessing as Pose (msg.position.x)
    if hasattr(msg, 'position'):
        return msg.position.x, msg.position.y, msg.position.z

    # 3. Try accessing as PoseStamped (msg.pose.position.x)
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
        return msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        
    return None, None, None

def read_rosbag2(bag_path, target_topics, max_duration_sec=50):
    bag_path = str(Path(bag_path).resolve())
    
    # 1. Setup Reader
    try:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening rosbag: {e}")
        return None

    # 2. Get Types
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    
    # 3. Prepare Storage
    data_store = {name: [] for name in target_topics.keys()}
    reverse_mapping = {v: k for k, v in target_topics.items()}
    
    start_time_ns = None
    max_duration_ns = max_duration_sec * 1e9
    
    print(f"Reading messages (max {max_duration_sec}s)...")
    
    # 4. Read Loop
    while reader.has_next():
        (topic, data, ts) = reader.read_next()
        
        if start_time_ns is None:
            start_time_ns = ts
            
        # Check duration limit
        if (ts - start_time_ns) > max_duration_ns:
            break

        if topic in reverse_mapping:
            name = reverse_mapping[topic]
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            
            x, y, z = extract_xyz(msg)
            
            if x is not None:
                # Use 'ts' (bag time) because Point msg has no header
                data_store[name].append({
                    'timestamp_ns': ts,
                    'timestamp_sec': (ts - start_time_ns) / 1e9, # Relative time
                    'point_x': x,
                    'point_y': y,
                    'point_z': z
                })

    # 5. Convert to DataFrames
    dataframes = {}
    for name, records in data_store.items():
        if records:
            dataframes[name] = pd.DataFrame(records)
            print(f"  {name}: {len(records)} records extracted")
        else:
            print(f"  Warning: No data found for {name} ({target_topics[name]})")
            
    return dataframes

def merge_dataframes(dataframes):
    """Merge leader and remote dataframes using nearest timestamp match"""
    if not dataframes or 'leader_pose' not in dataframes or 'remote_pose' not in dataframes:
        return None

    leader_df = dataframes['leader_pose'].sort_values('timestamp_ns')
    remote_df = dataframes['remote_pose'].sort_values('timestamp_ns')

    # Rename columns to avoid collisions
    leader_df = leader_df.rename(columns={
        'point_x': 'leader_x', 'point_y': 'leader_y', 'point_z': 'leader_z',
        'timestamp_sec': 'time_rel'
    })
    
    remote_df = remote_df.rename(columns={
        'point_x': 'remote_x', 'point_y': 'remote_y', 'point_z': 'remote_z'
    })

    # Merge using merge_asof (matches nearest timestamps)
    # We use leader timestamps as the base
    combined = pd.merge_asof(
        leader_df,
        remote_df[['timestamp_ns', 'remote_x', 'remote_y', 'remote_z']],
        on='timestamp_ns',
        direction='nearest',
        tolerance=int(100e6) # 100ms tolerance
    )
    
    # Clean up
    combined = combined.dropna()
    return combined

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_csv_fixed.py <bag_path>")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    output_file = Path(bag_path) / "teleoperation_data.csv"
    
    # 1. Read
    dfs = read_rosbag2(bag_path, TARGET_TOPICS, MAX_DURATION_SEC)
    
    # 2. Merge
    if dfs:
        combined_df = merge_dataframes(dfs)
        
        if combined_df is not None and not combined_df.empty:
            # 3. Save
            combined_df.to_csv(output_file, index=False)
            print(f"\nSuccess! Saved to: {output_file}")
            print(combined_df.head())
        else:
            print("Error: Could not merge data (timestamps might be too far apart or empty).")
    else:
        print("Error: No data extracted.")

if __name__ == '__main__':
    main()