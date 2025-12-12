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
import time # Added for timing

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped

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
    FIGURE_8 = "figure_8"
    SQUARE = "square"
    LISSAJOUS_COMPLEX = "lissajous_complex"

@dataclass(frozen=True)
class TrajectoryParams:
    center: NDArray[np.float64] = field(default_factory=lambda: TRAJECTORY_CENTER.copy())
    scale: NDArray[np.float64] = field(default_factory=lambda: TRAJECTORY_SCALE.copy())
    frequency: float = TRAJECTORY_FREQUENCY
    initial_phase: float = 0.0

    @classmethod
    def randomized(cls, actual_start_pos: NDArray[np.float64]) -> TrajectoryParams:
        center_x = np.random.uniform(0.3, 0.4)
        center_y = np.random.uniform(-0.1, 0.1)
        center_z = actual_start_pos[2]
        center = np.array([center_x, center_y, center_z], dtype=np.float64)
        scale_x = np.random.uniform(0.1, 0.3)  # Adjusted for variation, matching scale_y range
        scale_y = np.random.uniform(0.1, 0.3)
        scale_z = np.random.uniform(0.01, 0.03)  # Slight randomization for Z-scale
        scale = np.array([scale_x, scale_y, scale_z], dtype=np.float64)
        frequency = np.random.uniform(0.05, 0.15)
        return cls(center=center, scale=scale, frequency=frequency, initial_phase=0.0)

class TrajectoryGenerator(ABC):
    def __init__(self, params: TrajectoryParams):
        self._params = params
    @abstractmethod
    def compute_position(self, t: float) -> NDArray[np.float64]: pass
    def _compute_phase(self, t: float) -> float:
        return t * self._params.frequency * 2 * np.pi + self._params.initial_phase

class Figure8TrajectoryGenerator(TrajectoryGenerator):
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(3 * phase)
        dy = self._params.scale[1] * np.sin(4 * phase + np.pi / 2)
        dz = self._params.scale[2] * np.sin(phase)  # Use parameterized scale for consistency
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)

class SquareTrajectoryGenerator(TrajectoryGenerator):
    def compute_position(self, t: float) -> NDArray[np.float64]:
        period = 8.0
        phase = (t % period) / period * 4
        size = self._params.scale[0]
        if phase < 1: pos = [size, size * (phase), 0]
        elif phase < 2: pos = [size * (2 - phase), -size, 0]
        elif phase < 3: pos = [-size, -size * (phase - 2), 0]
        else: pos = [-size * (4 - phase), size, 0]
        return self._params.center + np.array(pos, dtype=np.float64)

class LissajousTrajectoryGenerator(TrajectoryGenerator):
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(3 * phase)
        dy = self._params.scale[1] * np.sin(4 * phase + np.pi / 2)
        dz = 0.02 * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)

class LeaderRobotPublisher(Node):
    def __init__(self):
        super().__init__('leader_robot_publisher')
        self.publish_freq = DEFAULT_PUBLISH_FREQ
        self._dt = 1.0 / self.publish_freq

        traj_type_str = self.declare_parameter('trajectory_type', TrajectoryType.FIGURE_8.value).value
        self._trajectory_type = TrajectoryType(traj_type_str)
        self._randomize_params = self.declare_parameter('randomize_params', False).value
        self.model_path = self.declare_parameter('model_path', DEFAULT_MUJOCO_MODEL_PATH).value

        self.n_joints = N_JOINTS
        self.ee_body_name = EE_BODY_NAME
        self.tcp_offset = TCP_OFFSET.copy()
        self.joint_limits_lower = JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = JOINT_LIMITS_UPPER.copy()
        self.joint_names = [f'panda_joint{i+1}' for i in range(self.n_joints)]

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.ik_solver = IKSolver(self.model, self.joint_limits_lower, self.joint_limits_upper)

        # 1. Get Actual Spawn Position (Home FK)
        self._q_start = INITIAL_JOINT_CONFIG.copy()
        self.data.qpos[:self.n_joints] = self._q_start
        mujoco.mj_forward(self.model, self.data)
        ee_site_id = self.model.site('panda_ee_site').id
        self.actual_spawn_pos = self.data.site_xpos[ee_site_id].copy()
        
        self.get_logger().info(f"Start EE position (FK): {self.actual_spawn_pos}")

        if self._randomize_params:
            self._params = TrajectoryParams.randomized(self.actual_spawn_pos)
        else:
            self._params = TrajectoryParams()

        self._generator = self._create_generator(self._trajectory_type, self._params)
        self.traj_start_pos = self._generator.compute_position(0.0)

        self._trajectory_time = 0.0
        self._tick = 0
        self._q_current = self._q_start.copy()
        self._q_previous = self._q_start.copy()

        self.ik_solver.reset_trajectory(q_start=self._q_start)

        self.joint_state_pub = self.create_publisher(JointState, 'local_robot/joint_states', 100)
        self.ee_pose_pub = self.create_publisher(PointStamped, 'local_robot/ee_pose', 100)
        
        self.timer = self.create_timer(self._dt, self.timer_callback)
        self.start_wall_time = time.time() # Debug
        
        self.get_logger().info("Leader Robot Publisher started (Interpolated Warmup).")

    def _create_generator(self, trajectory_type: TrajectoryType, params: TrajectoryParams) -> TrajectoryGenerator:
        generators = {
            TrajectoryType.FIGURE_8: Figure8TrajectoryGenerator,
            TrajectoryType.SQUARE: SquareTrajectoryGenerator,
            TrajectoryType.LISSAJOUS_COMPLEX: LissajousTrajectoryGenerator,
        }
        return generators[trajectory_type](params)

    def timer_callback(self) -> None:
        self._trajectory_time += self._dt
        self._tick += 1
        t = self._trajectory_time

        # --- FIX: Linear Interpolation during Warmup ---
        if t < WARM_UP_DURATION:
            progress = t / WARM_UP_DURATION
            current_target_pos = (1 - progress) * self.actual_spawn_pos + progress * self.traj_start_pos
            q_target_raw, ik_success, _ = self.ik_solver.solve(current_target_pos, self._q_current)
        else:
            movement_time = t - WARM_UP_DURATION
            cartesian_target = self._generator.compute_position(movement_time)
            q_target_raw, ik_success, _ = self.ik_solver.solve(cartesian_target, self._q_current)
            
        if not ik_success or q_target_raw is None:
            # self.get_logger().warn(f"IK failed at t={t:.3f}s. Holding last q.")
            q_target_raw = self._q_current.copy()

        qd_raw = (q_target_raw - self._q_previous) / self._dt
        self._q_previous = self._q_current.copy()
        self._q_current = q_target_raw.copy()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.name = self.joint_names
        msg.position = self._q_current.astype(float).tolist()
        msg.velocity = qd_raw.astype(float).tolist()
        
        self.joint_state_pub.publish(msg)

        # Publish EE Pose
        self.data.qpos[:self.n_joints] = self._q_current
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
        
        # --- DEBUG PRINT ---
        if self._tick % 20 == 0: # Print every 20 steps (10Hz)
            print(f"[Leader] Tick: {self._tick} | Time: {t:.3f}s | EE_Z: {ee_pos[2]:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = LeaderRobotPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()