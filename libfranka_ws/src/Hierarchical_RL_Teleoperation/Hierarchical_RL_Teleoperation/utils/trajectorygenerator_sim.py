"""
Trajectory Generator Node for Publishing a Continuous Figure-8 Trajectory.

This version starts the dynamic trajectory immediately for robust testing.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import time

class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('trajectory_generator')
        
        # --- Parameters ---
        self.publish_freq = 200  # Hz
        self.traj_center = np.array([0.4, 0.0, 0.5])
        self.traj_scale = np.array([0.15, 0.25, 0.15]) # Increased scale for more dynamic motion
        self.traj_freq = 0.15  # Hz, slightly faster

        # --- ROS Interfaces ---
        self.publisher = self.create_publisher(PoseStamped, '/local_robot/leader_pose', 10)
        self.timer = self.create_timer(1.0 / self.publish_freq, self.publish_trajectory_point)
        
        # --- Trajectory State ---
        self.trajectory_start_time = time.time()
        self.time_step = 0
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("DIRECT TRAJECTORY GENERATOR (FIGURE-8)")
        self.get_logger().info(f"Publishing to '/local_robot/leader_pose' at {self.publish_freq} Hz.")
        self.get_logger().info("=" * 60)

    def publish_trajectory_point(self):
        """
        Calculates and publishes the next point in the figure-8 trajectory.
        """
        current_time = time.time() - self.trajectory_start_time
        
        # --- Generate Trajectory Point ---
        t = current_time * self.traj_freq * 2 * np.pi
        
        dx = self.traj_scale[0] * np.sin(t)
        dy = self.traj_scale[1] * np.sin(t / 2)
        dz = self.traj_scale[2] * np.cos(t) * 0.5 # Add some vertical motion
        
        position = self.traj_center + np.array([dx, dy, dz])
        
        # --- Publish Pose ---
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "panda_link0"
        
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        
        # Default orientation (pointing downwards)
        msg.pose.orientation.x = 1.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 0.0

        self.publisher.publish(msg)
        
        # Log periodically
        if self.time_step % (self.publish_freq * 2) == 0: # Log every 2 seconds
            self.get_logger().info(
                f'Published point: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]'
            )
        
        self.time_step += 1


def main(args=None):
    rclpy.init(args=args)
    trajectory_publisher = None
    try:
        trajectory_publisher = TrajectoryGenerator()
        rclpy.spin(trajectory_publisher)
    except KeyboardInterrupt:
        print("\nTrajectory publishing stopped by user.")
    except Exception as e:
        print(f"\nAn unrecoverable error occurred: {e}")
    finally:
        if trajectory_publisher:
            trajectory_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Trajectory publisher shutdown complete.")

if __name__ == '__main__':
    main()
