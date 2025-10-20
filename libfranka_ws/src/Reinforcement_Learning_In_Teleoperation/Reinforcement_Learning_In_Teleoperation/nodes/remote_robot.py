"""
Remote robot node using ROS2 (Follower).

The node implements:
- Subscribes to real robot's joint state
- Subscribes to desired robot's joint state
- execute Inverse dynamics + PD control law
- implement RL tau compensation
- Publishes the command to the real robot
"""

import rclpy
from rclpy.node import Node

