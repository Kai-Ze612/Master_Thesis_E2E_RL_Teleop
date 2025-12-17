"""
Trajectory Generator Node for Publishing a Sequence of Discrete Points
to /local_robot/leader_pose.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np

class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('point_cycler_publisher')
        
        self._init_parameters()
        self._init_ros_interfaces()

        # Use the ROS clock for all timing operations
        self.start_time = self.get_clock().now()
        
        self.get_logger().info("Point Cycler Publisher node initialized.")

    def _init_parameters(self):
        """Initialize all parameters for the node and the point sequence."""
        self.publish_freq = 200  # Hz for the publisher timer

        ### MODIFICATION 1: Parameters for cyclic motion ###
        # Define the two points for the back-and-forth motion
        self.point_a = np.array([0.3, 0.0, 0.6])
        self.point_b = np.array([0.4, 0.0, 0.6])
        
        # Define how long to wait at EACH point before switching
        self.dwell_time = 5.0 # seconds

    def _init_ros_interfaces(self):
        """Initialize ROS publishers and timers."""
        self.publisher = self.create_publisher(
            PoseStamped, '/local_robot/leader_pose', 10
        )
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_point)
    
    def publish_point(self):
        """Calculates and publishes the current target point based on a repeating cycle."""
        
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        ### MODIFICATION 2: Logic for a repeating back-and-forth cycle ###
        # A full cycle is moving to B and back to A
        cycle_period = 2 * self.dwell_time
        time_in_cycle = elapsed_time % cycle_period
        
        if time_in_cycle < self.dwell_time:
            # First half of the cycle: publish Point A
            current_target_position = self.point_a
        else:
            # Second half of the cycle: publish Point B
            current_target_position = self.point_b
            
        # Create and populate the PoseStamped message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "panda_link0"
        
        msg.pose.position.x = current_target_position[0]
        msg.pose.position.y = current_target_position[1]
        msg.pose.position.z = current_target_position[2]
        
        # A standard orientation for pointing the tool downwards
        msg.pose.orientation.x = 1.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 0.0

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    point_publisher = None
    try:
        point_publisher = TrajectoryGenerator()
        ### MODIFICATION 3: Updated print statements for the new logic ###
        print("=" * 50)
        print("--- Point Cycler Publisher Running ---")
        print(f"  Moving between Point A: {point_publisher.point_a}")
        print(f"  and Point B: {point_publisher.point_b}")
        print(f"  Waiting at each point for: {point_publisher.dwell_time} seconds")
        print("Press Ctrl+C to stop.")
        print("=" * 50)
        rclpy.spin(point_publisher)
        
    except KeyboardInterrupt:
        pass 
    finally:
        if point_publisher:
            point_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("\n--- Point Publisher Shutdown Complete ---")

if __name__ == '__main__':
    main()