"""
Trajectory Generator Node for Publishing Continuous Figure-8 Trajectory Points

to /local_robot/leader_pose.

"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import time

class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('single_cycle_trajectory_publisher')
        
        self.publish_freq = 500
        self.timer = self.create_timer(1.0/self.publish_freq, self.publish_trajectory_point)
        
        self._init_ROS_interfaces()
        
        # --- Trajectory Parameters ---
        self.traj_center = np.array([0.4, 0.0, 0.6])
        self.traj_scale = np.array([0.1, 0.3, 0.0])
        self.traj_freq = 0.1  # Hz

        self.cycle_duration = 2.0 / self.traj_freq

        # Define the states of the generator
        self.State = {
            'INITIALIZING': 0,
            'RUNNING': 1,
            'FINISHED': 2 
        }
        self.current_state = self.State['INITIALIZING']

        # Define the initialization sequence
        self.init_points = [
            np.array([0.4, 0.0, 0.6])
        ]
        
        self.init_phase = 0
        self.wait_duration = 5.0  # seconds
        self.wait_start_time = time.time()
        self.get_logger().info(f"Starting initialization: Moving to point {self.init_phase + 1}/{len(self.init_points)}: {self.init_points[self.init_phase]}")

        # Trajectory state
        self.time_step = 0
        self.trajectory_start_time = 0

    def _init_ROS_interfaces(self):
        """Initialize ROS interfaces."""
        self.publisher = self.create_publisher(
            PoseStamped, '/local_robot/leader_pose', 10
        )
    
    def publish_trajectory_point(self):
        """
        State machine dispatcher.
        Publishes a point based on the current state.
        """
        if self.current_state == self.State['INITIALIZING']:
            self.handle_initialization()
        elif self.current_state == self.State['RUNNING']:
            self.handle_running()
        # --- MODIFICATION: Handle the FINISHED state to shut down ---
        elif self.current_state == self.State['FINISHED']:
            self.get_logger().info("Trajectory complete. Shutting down publisher node.")
            self.timer.cancel()
            self.destroy_node() # This will cause rclpy.spin() to exit

    def handle_initialization(self):
        """
        Publishes static points and transitions to RUNNING state.
        """
        position = self.init_points[self.init_phase]
        self.publish_pose(position)

        if time.time() - self.wait_start_time >= self.wait_duration:
            self.init_phase += 1
            
            if self.init_phase >= len(self.init_points):
                self.get_logger().info("=" * 30)
                self.get_logger().info(f"Initialization complete. Starting single figure-8 trajectory ({self.cycle_duration:.1f}s).")
                self.get_logger().info("=" * 30)
                self.current_state = self.State['RUNNING']
                self.trajectory_start_time = time.time()
            else:
                self.get_logger().info(f"Holding complete. Moving to point {self.init_phase + 1}/{len(self.init_points)}: {self.init_points[self.init_phase]}")
                self.wait_start_time = time.time()

    def handle_running(self):
        """
        --- MODIFICATION: Checks if the cycle is complete before publishing. ---
        Publishes trajectory points until the cycle duration is met.
        """
        current_time = time.time() - self.trajectory_start_time
        
        # Only publish if we are within the single cycle duration
        if current_time <= self.cycle_duration:
            position = self.generate_figure8_trajectory(current_time)
            self.publish_pose(position)

            if self.time_step % 500 == 0:
                self.get_logger().info(
                    f'Published point {self.time_step}: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] at t={current_time:.2f}s'
                )
            
            self.time_step += 1
        else:
            # Transition to the FINISHED state if the cycle is done
            self.get_logger().info("=" * 30)
            self.get_logger().info(f"Figure-8 trajectory cycle complete after {current_time:.2f} seconds.")
            self.get_logger().info("=" * 30)
            self.current_state = self.State['FINISHED']

    def publish_pose(self, position_vec):
        """
        Creates and publishes a pose with a given position and default orientation.
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "panda_link0"
        
        msg.pose.position.x = float(position_vec[0])
        msg.pose.position.y = float(position_vec[1])
        msg.pose.position.z = float(position_vec[2])
        
        msg.pose.orientation.x = 1.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 0.0

        self.publisher.publish(msg)
    
    def generate_figure8_trajectory(self, traj_time):
        """
        Generate figure-8 trajectory based on the provided time.
        """
        t = traj_time * self.traj_freq * 2 * np.pi
        
        dx = self.traj_scale[0] * np.sin(t)
        dy = self.traj_scale[1] * np.sin(t / 2)
        dz = 0.0
        
        target_position = self.traj_center + np.array([dx, dy, dz])
        
        return target_position

def main(args=None):
    rclpy.init(args=args)
    
    trajectory_publisher = None
    try:
        trajectory_publisher = TrajectoryGenerator()
        print("=" * 60)
        print("TRAJECTORY GENERATOR - SINGLE FIGURE-8 CYCLE")
        print("=" * 60)
        print(f"Executing initialization sequence...")
        print(f"Publishing Frequency: {trajectory_publisher.publish_freq} Hz")
        print(f"Trajectory will run for one cycle ({trajectory_publisher.cycle_duration:.1f} seconds) and then exit.")
        print("=" * 60)

        rclpy.spin(trajectory_publisher)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Trajectory publishing stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # The node now handles its own destruction, but this is good practice
        if trajectory_publisher and not trajectory_publisher.is_destroyed():
            trajectory_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("=" * 60)
        print("ROS Shutdown complete.")
        print("=" * 60)

if __name__ == '__main__':
    main()
