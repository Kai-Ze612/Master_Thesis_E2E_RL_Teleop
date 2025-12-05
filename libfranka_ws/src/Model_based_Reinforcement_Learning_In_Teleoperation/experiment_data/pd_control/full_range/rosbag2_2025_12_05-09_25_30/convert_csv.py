#!/usr/bin/env python3
"""
Fixed rosbag2 to CSV converter that handles sqlite3 rosbag2 format correctly.
"""

import sys
from pathlib import Path
import pandas as pd

try:
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    sys.exit(1)


def message_to_dict(msg):
    """Convert ROS message to dictionary."""
    result = {}
    if not hasattr(msg, 'get_fields_and_field_types'):
        return result
    
    for field_name, field_type in msg.get_fields_and_field_types().items():
        value = getattr(msg, field_name)
        if hasattr(value, 'get_fields_and_field_types'):
            result[field_name] = message_to_dict(value)
        elif isinstance(value, (list, tuple)):
            result[field_name] = str(value)
        else:
            result[field_name] = value
    
    return result


def extract_pose_data(msg_dict):
    """Extract timestamp and point coordinates."""
    try:
        timestamp_sec = msg_dict.get('header', {}).get('stamp', {}).get('sec', None)
        x = msg_dict.get('point', {}).get('x', None)
        y = msg_dict.get('point', {}).get('y', None)
        z = msg_dict.get('point', {}).get('z', None)
        
        return {
            'timestamp_sec': timestamp_sec,
            'point_x': x,
            'point_y': y,
            'point_z': z
        }
    except Exception as e:
        return None


def read_rosbag2_sqlite3(bag_path, target_topics, max_duration_sec=50):
    """
    Read rosbag2 stored in sqlite3 format.
    
    Args:
        bag_path: Path to rosbag2 directory or parent directory
        target_topics: Dict mapping {name: topic_path}
        max_duration_sec: Maximum duration in seconds to extract
    
    Returns:
        Dict mapping {name: list of dicts}
    """
    bag_path = Path(bag_path).resolve()
    
    # If the path is the rosbag2 directory, use it; otherwise find the right directory
    if not (bag_path / 'metadata.yaml').exists():
        print(f"metadata.yaml not found in {bag_path}")
        print("Searching for rosbag2 directory...")
        return None
    
    print(f"Opening rosbag2 from: {bag_path}")
    
    try:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(bag_path),
            storage_id='sqlite3'
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening rosbag2: {e}")
        return None
    
    # Get topic information
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    
    print(f"\nFound {len(type_map)} topics:")
    for topic_name, topic_type in sorted(type_map.items()):
        print(f"  {topic_name}: {topic_type}")
    
    # Data extraction
    data_store = {name: [] for name in target_topics.keys()}
    reverse_mapping = {v: k for k, v in target_topics.items()}
    
    message_count = 0
    start_time_ns = None
    max_duration_ns = max_duration_sec * 1e9
    
    print(f"\nExtracting messages (max {max_duration_sec} seconds)...")
    
    try:
        while reader.has_next():
            topic, data, ts = reader.read_next()
            
            # Set start time on first message
            if start_time_ns is None:
                start_time_ns = ts
            
            # Check duration limit
            elapsed_ns = ts - start_time_ns
            if elapsed_ns > max_duration_ns:
                print(f"Reached {max_duration_sec} second limit")
                break
            
            if topic in reverse_mapping:
                name = reverse_mapping[topic]
                
                try:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    
                    msg_dict = message_to_dict(msg)
                    pose_data = extract_pose_data(msg_dict)
                    
                    if pose_data:
                        pose_data['timestamp_ns'] = ts
                        data_store[name].append(pose_data)
                    
                    message_count += 1
                    
                    if message_count % 1000 == 0:
                        elapsed_sec = elapsed_ns / 1e9
                        print(f"  Processed {message_count} messages ({elapsed_sec:.2f}s)")
                
                except Exception as e:
                    pass
    
    except Exception as e:
        print(f"Error reading messages: {e}")
        return None
    
    print(f"Total messages extracted: {message_count}")
    return data_store


def write_to_csv(data_store, output_csv):
    """Write extracted data to CSV."""
    output_file = Path(output_csv).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrames
    leader_df = None
    follower_df = None
    
    if data_store.get('local_pose'):
        leader_df = pd.DataFrame(data_store['local_pose'])
        leader_df = leader_df.sort_values('timestamp_ns').reset_index(drop=True)
        leader_df = leader_df.rename(columns={
            'timestamp_sec': 'leader_ee_pose_header.stamp.sec',
            'point_x': 'leader_ee_pose_point.x',
            'point_y': 'leader_ee_pose_point.y',
            'point_z': 'leader_ee_pose_point.z'
        })
    
    if data_store.get('remote_pose'):
        follower_df = pd.DataFrame(data_store['remote_pose'])
        follower_df = follower_df.sort_values('timestamp_ns').reset_index(drop=True)
        follower_df = follower_df.rename(columns={
            'timestamp_sec': 'follower_ee_pose_header.stamp.sec',
            'point_x': 'follower_ee_pose_point.x',
            'point_y': 'follower_ee_pose_point.y',
            'point_z': 'follower_ee_pose_point.z'
        })
    
    if leader_df is None:
        print("Error: No local pose data")
        return False
    
    if follower_df is None:
        print("Warning: No remote pose data, saving local only")
        leader_df = leader_df.drop(columns=['timestamp_ns'])
        leader_df.to_csv(output_file, index=False)
        print(f"CSV created: {output_file}")
        return True
    
    # Merge on timestamp
    df_combined = pd.merge_asof(
        leader_df, follower_df,
        on='timestamp_ns',
        direction='nearest',
        tolerance=int(100e6)
    )
    df_combined = df_combined.drop(columns=['timestamp_ns'])
    df_combined.to_csv(output_file, index=False)
    
    print(f"\nCSV created successfully: {output_file}")
    print(f"Total rows: {len(df_combined)}")
    print(f"Columns: {list(df_combined.columns)}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 rosbag2_converter_fixed.py <rosbag2_path> [output_csv]")
        print("\nExample:")
        print("  python3 rosbag2_converter_fixed.py /path/to/rosbag2_2025_12_05-09_23_44")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else str(Path(bag_path).parent / "teleoperation_combined.csv")
    
    # Define topics - note: your rosbag2 uses different topic names
    target_topics = {
        'local_pose': '/local_robot/ee_pose',
        'remote_pose': '/remote_robot/ee_pose'
    }
    
    print("=== ROS2 Bag to CSV Converter ===\n")
    
    data_store = read_rosbag2_sqlite3(bag_path, target_topics, max_duration_sec=50)
    
    if data_store:
        success = write_to_csv(data_store, output_csv)
        if success:
            print("\nConversion completed successfully!")
            sys.exit(0)
    
    print("\nConversion failed")
    sys.exit(1)


if __name__ == '__main__':
    main()