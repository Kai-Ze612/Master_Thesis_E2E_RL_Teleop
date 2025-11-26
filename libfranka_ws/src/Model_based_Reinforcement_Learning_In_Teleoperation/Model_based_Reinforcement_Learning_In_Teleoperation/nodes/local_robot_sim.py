"""
The script is the local robot, using ROS2 node. This is a virtual trajectory generator.

Pipeline:
1. trajectory generation in Cartesian space (with Z-axis Sine Wave)
2. inverse kinematics to get joint space commands
3. publish joint states to /local_robot/joint_states topic
"""


from __future__ import annotations 

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from numpy.typing import NDArray
import mujoco

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped

# Custom imports
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.inverse_kinematics import IKSolver
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    DEFAULT_MUJOCO_MODEL_PATH,
    N_JOINTS,
    EE_BODY_NAME,
    TCP_OFFSET,
    INITIAL_JOINT_CONFIG,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    DEFAULT_PUBLISH_FREQ,
    TRAJECTORY_CENTER,
    TRAJECTORY_SCALE,
    TRAJECTORY_FREQUENCY,
    WARM_UP_DURATION,
)


class TrajectoryType(Enum):
    """Enumeration of available trajectory types."""
    FIGURE_8 = "figure_8"
    SQUARE = "square"
    LISSAJOUS_COMPLEX = "lissajous_complex"


@dataclass(frozen=True)
class TrajectoryParams:
    """Trajectory parameters - matches RL training."""
    center: NDArray[np.float64] = field(
        default_factory=lambda: TRAJECTORY_CENTER.copy()
    )
    scale: NDArray[np.float64] = field(
        default_factory=lambda: TRAJECTORY_SCALE.copy()
    )
    frequency: float = TRAJECTORY_FREQUENCY
    initial_phase: float = 0.0

    @classmethod
    def randomized(cls, actual_start_pos: NDArray[np.float64]) -> TrajectoryParams:
        """Randomize trajectory parameters based on actual start position."""
        # Randomize Center
        center_x = np.random.uniform(0.3, 0.4)
        center_y = np.random.uniform(-0.1, 0.1)
        center_z = actual_start_pos[2]
        
        center = np.array([center_x, center_y, center_z], dtype=np.float64)
        
        # Randomize Scale
        scale_x = np.random.uniform(0.1, 0.1)  # Kept as provided in simulator (0.1 to 0.1)
        scale_y = np.random.uniform(0.1, 0.3)
        scale_z = 0.02
        
        scale = np.array([scale_x, scale_y, scale_z], dtype=np.float64)
        
        # Randomize Frequency
        frequency = np.random.uniform(0.05, 0.15)
        
        return cls(center=center, scale=scale, frequency=frequency, initial_phase=0.0)


class TrajectoryGenerator(ABC):
    """Base class for trajectory generators."""
    
    def __init__(self, params: TrajectoryParams):
        self._params = params

    @abstractmethod
    def compute_position(self, t: float) -> NDArray[np.float64]:
        """Compute Cartesian position at time t."""
        pass
    
    def _compute_phase(self, t: float) -> float:
        """Compute phase angle at time t."""
        return t * self._params.frequency * 2 * np.pi + self._params.initial_phase


class Figure8TrajectoryGenerator(TrajectoryGenerator):
    """Figure-8 trajectory using Lissajous curve with 1:2 frequency ratio."""
    
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(phase)
        dy = self._params.scale[1] * np.sin(phase / 2)
        dz = self._params.scale[2] * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)


class SquareTrajectoryGenerator(TrajectoryGenerator):
    """Square trajectory logic aligned with local_robot_simulator.py."""
    
    def compute_position(self, t: float) -> NDArray[np.float64]:
        period = 8.0
        phase = (t % period) / period * 4
        size = self._params.scale[0]
        
        if phase < 1:
            pos = [size, size * (phase), 0]
        elif phase < 2:
            pos = [size * (2 - phase), -size, 0]
        elif phase < 3:
            pos = [-size, -size * (phase - 2), 0]
        else:
            pos = [-size * (4 - phase), size, 0]
            
        return self._params.center + np.array(pos, dtype=np.float64)


class LissajousTrajectoryGenerator(TrajectoryGenerator):
    """Complex Lissajous curve with 3:4 frequency ratio."""
    
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(3 * phase)
        dy = self._params.scale[1] * np.sin(4 * phase + np.pi / 2)
        dz = 0.02 * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)


class LeaderRobotPublisher(Node):
    """
    ROS2 Node - Pure Kinematic Trajectory Generator.
    
    Matches the behavior of LocalRobotSimulator used in RL training.
    NO MuJoCo physics simulation - only trajectory generation + IK.
    """
    
    def __init__(self):
        super().__init__('leader_robot_publisher')

        # ROS2 parameters
        self.publish_freq = DEFAULT_PUBLISH_FREQ
        self._dt = 1.0 / self.publish_freq

        # Trajectory type parameter
        traj_type_str = self.declare_parameter(
            'trajectory_type', TrajectoryType.FIGURE_8.value
        ).value
        self._trajectory_type = TrajectoryType(traj_type_str)
        
        self._randomize_params = self.declare_parameter('randomize_params', False).value
        self.model_path = self.declare_parameter('model_path', DEFAULT_MUJOCO_MODEL_PATH).value

        # Robot parameters
        self.n_joints = N_JOINTS
        self.ee_body_name = EE_BODY_NAME
        self.tcp_offset = TCP_OFFSET.copy()
        self.joint_limits_lower = JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = JOINT_LIMITS_UPPER.copy()
        
        # Joint names for ROS2 messages
        self.joint_names = [f'panda_joint{i+1}' for i in range(self.n_joints)]

        # Load MuJoCo model for IK/FK ONLY (no physics simulation)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Initialize IK solver
        self.ik_solver = IKSolver(
            model=self.model,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper,
        )

        # Get actual start position via FK
        self._q_start = INITIAL_JOINT_CONFIG.copy()
        self.data.qpos[:self.n_joints] = self._q_start
        mujoco.mj_forward(self.model, self.data)
        
        ee_site_id = self.model.site('panda_ee_site').id
        actual_start_pos = self.data.site_xpos[ee_site_id].copy()
        
        self.get_logger().info(f"Start EE position (FK): {actual_start_pos}")

        # Initialize trajectory generator with randomization logic
        if self._randomize_params:
            self._params = TrajectoryParams.randomized(actual_start_pos)
            self.get_logger().info("Trajectory parameters Randomized.")
        else:
            self._params = TrajectoryParams(center=actual_start_pos.copy())
            self.get_logger().info("Using default Trajectory parameters.")

        self._generator = self._create_generator(self._trajectory_type, self._params)

        # State tracking (kinematic only)
        self._trajectory_time = 0.0
        self._tick = 0
        self._q_current = self._q_start.copy()
        self._q_previous = self._q_start.copy()

        # Reset IK solver
        self.ik_solver.reset_trajectory(q_start=self._q_start)

        # Create ROS2 publishers
        self.joint_state_pub = self.create_publisher(
            JointState, 'local_robot/joint_states', 100
        )
        self.ee_pose_pub = self.create_publisher(
            PointStamped, 'local_robot/ee_pose', 100
        )
        
        # Create timer
        self.timer = self.create_timer(self._dt, self.timer_callback)

        self.get_logger().info(
            f"Leader Robot Publisher started.\n"
            f"  Trajectory: {self._trajectory_type.value}\n"
            f"  Frequency: {self.publish_freq} Hz\n"
            f"  Warm-up: {WARM_UP_DURATION} s"
        )

    def _create_generator(
        self,
        trajectory_type: TrajectoryType,
        params: TrajectoryParams,
    ) -> TrajectoryGenerator:
        """Create trajectory generator based on type."""
        generators = {
            TrajectoryType.FIGURE_8: Figure8TrajectoryGenerator,
            TrajectoryType.SQUARE: SquareTrajectoryGenerator,
            TrajectoryType.LISSAJOUS_COMPLEX: LissajousTrajectoryGenerator,
        }
        
        generator_class = generators.get(trajectory_type)
        if generator_class is None:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        
        return generator_class(params)

    def timer_callback(self) -> None:
        """
        Timer callback - generates and publishes trajectory.
        
        Matches LocalRobotSimulator.step() behavior.
        """
        self._trajectory_time += self._dt
        self._tick += 1
        t = self._trajectory_time

        # Generate Cartesian target
        if t < WARM_UP_DURATION:
            # Hold at start during warm-up
            q_desired = self._q_start.copy()
        else:
            # Follow trajectory
            movement_time = t - WARM_UP_DURATION
            cartesian_target = self._generator.compute_position(movement_time)
            
            # Solve IK
            q_desired, ik_success, ik_error = self.ik_solver.solve(
                cartesian_target, self._q_current
            )
            
            if not ik_success or q_desired is None:
                self.get_logger().warn(
                    f"IK failed at t={t:.3f}s, error={ik_error:.6f}m. Using last q."
                )
                q_desired = self._q_current.copy()

        # Compute velocity via finite difference
        qd_desired = (q_desired - self._q_previous) / self._dt

        # Update state
        self._q_previous = self._q_current.copy()
        self._q_current = q_desired.copy()

        # Publish joint states
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.name = self.joint_names
        msg.position = q_desired.astype(float).tolist()
        msg.velocity = qd_desired.astype(float).tolist()
        msg.effort = []
        
        self.joint_state_pub.publish(msg)

        # Publish EE pose (via FK)
        self.data.qpos[:self.n_joints] = q_desired
        mujoco.mj_forward(self.model, self.data)
        ee_site_id = self.model.site('panda_ee_site').id
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        
        ee_msg = PointStamped()
        ee_msg.header.stamp = msg.header.stamp
        ee_msg.header.frame_id = "world"
        ee_msg.point.x = float(ee_pos[0])
        ee_msg.point.y = float(ee_pos[1])
        ee_msg.point.z = float(ee_pos[2])
        
        self.ee_pose_pub.publish(ee_msg)

    def reset(self) -> None:
        """Reset trajectory generator to initial state."""
        self._trajectory_time = 0.0
        self._tick = 0
        self._q_current = self._q_start.copy()
        self._q_previous = self._q_start.copy()
        self.ik_solver.reset_trajectory(q_start=self._q_start)
        
        self.get_logger().info("Trajectory reset to initial state.")


def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = LeaderRobotPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Node failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()