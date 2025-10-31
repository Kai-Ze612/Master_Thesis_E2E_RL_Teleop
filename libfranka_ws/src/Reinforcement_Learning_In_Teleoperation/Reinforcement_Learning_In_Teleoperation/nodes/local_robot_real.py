"""
The script is the local robot, using ROS2 node. This is a real robot arm.

Human operator controls the local robot arm directly by moving it, then we read the joint states from the real robot and publish them to /local_robot/joint_states topic for the remote robot to execute.
"""

# Python imports
import numpy as np
import torch
from collections import deque
import sys

# ROS2 Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# custom imports
from Reinforcement_Learning_In_Teleoperation.config.robot_config import N_JOINTS

class LocalRobotReal(Node):
    
    def __init__(self):
        super().__init__('local_robot_real')

        # Initialize ROS2 topic
        self.driver_topic_ = self.declare_parameter(
            'driver_topic', '/local_robot/joint_states').value # Reading current robot joint states from real robot
        
        self.publish_topic_ = self.declare_parameter(
            'publish_topic', '/local_robot/joint_states').value
        
        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(N_JOINTS)]
        
        # ROS2 interfaces
        self.publisher_ = self.create_publisher(
            JointState,
            self.publish_topic_,
            10
        )
        
        self.subscription_ = self.create_subscription(
            JointState,
            self.driver_topic_,
            self.driver_callback,
            10
        )

    def driver_callback(self, msg: JointState) -> None:
        # Process incoming joint state messages from the real robot

        # Create a mapping from incoming message's joint names to their data indices
        name_to_index_map = {name: i for i, name in enumerate(msg.name)}
        
        new_positions = []
        new_velocities = []
        new_efforts = []
        
        for target_name in self.target_joint_names_:
            # Find the index of this joint in the *incoming* message
            incoming_index = name_to_index_map[target_name]
            
            # Append the data from that index
            new_positions.append(msg.position[incoming_index])
            new_velocities.append(msg.velocity[incoming_index])
            # Handle cases where effort might not be published
            if msg.effort:
                new_efforts.append(msg.effort[incoming_index])
                
        new_msg = JointState()
        new_msg.header.stamp = self.get_clock().now().to_msg()
        new_msg.name = self.target_joint_names_
        new_msg.position = new_positions
        new_msg.velocity = new_velocities
        new_msg.effort = new_efforts
        
        self.publisher_.publish(new_msg)
        
def main(args=None):
    rclpy.init(args=args)
    real_leader_node = None
    
    try:
        real_leader_node = LocalRobotReal()
        rclpy.spin(real_leader_node)
    except KeyboardInterrupt:
        real_leader_node.get_logger().info("Keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Node failed to initialize or run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if real_leader_node:
            real_leader_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()