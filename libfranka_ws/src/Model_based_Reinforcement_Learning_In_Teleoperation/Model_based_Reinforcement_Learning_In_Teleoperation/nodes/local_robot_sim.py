"""
The script is the local robot, using ROS2 node. This is a virtual trajectory generator.

Pipeline:
1. trajectory generation in Cartesian space (with Z-axis Sine Wave)
2. inverse kinematics to get joint space commands
3. publish joint states to /local_robot/joint_states topic
"""

# python imports
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
    IK_MAX_ITER,
    IK_TOLERANCE,
    IK_DAMPING,
    IK_MAX_JOINT_CHANGE,
    IK_CONTINUITY_GAIN,
    IK_STEP_SIZE,
    TRAJECTORY_CENTER,
    TRAJECTORY_SCALE,
    TRAJECTORY_FREQUENCY,
    WARM_UP_DURATION,
)

Z_AMPLITUDE = 0.01

class TrajectoryType(Enum):
    """Enumeration for different trajectory types."""
    FIGURE_8 = "figure_8"
    SQUARE = "square"
    LISSAJOUS_COMPLEX = "lissajous_complex"

@dataclass(frozen=True)
class TrajectoryParameters:
    """Trajectory initial parameters."""
    center: NDArray[np.float64] = field(
        default_factory=lambda: TRAJECTORY_CENTER.copy()
    )
    scale: NDArray[np.float64] = field(
        default_factory=lambda: TRAJECTORY_SCALE.copy()
    )
    frequency: float = TRAJECTORY_FREQUENCY
    initial_phase: float = 0.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        assert self.center.shape == (3,), "Center must be a 3D point."
        assert self.scale.shape == (2,), "Scale must be a 2D vector."
        assert self.frequency > 0, "Frequency must be positive."
        
class TrajectoryGenerator(ABC):
    """Position computation"""
    
    def __init__(self,
                 params: TrajectoryParameters):
        self._params = params
        
    @property
    def params(self) -> TrajectoryParameters:
        return self._params

    @abstractmethod
    def compute_position(self, t: float) -> NDArray[np.float64]:
        """Compute position at time t."""
        pass
    
    def _compute_phase(self, t: float) -> float:
        """Compute phase angle at time t."""
        return (t * self._params.frequency * 2 * np.pi + 
                self._params.initial_phase)

class SquareTrajectoryGenerator(TrajectoryGenerator):
    """Square trajectory in XY plane with smooth corners."""
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        t_norm = (phase % (2 * np.pi)) / (2 * np.pi)
        
        corners = np.array([
            [1, 1], [-1, 1], [-1, -1], [1, -1],
        ])
        
        segment = int(t_norm * 4) % 4
        segment_progress = (t_norm * 4) % 1
        
        current_corner = corners[segment]
        next_corner = corners[(segment + 1) % 4]
        
        smooth_progress = 0.5 * (1 - np.cos(segment_progress * np.pi))
        position_2d = current_corner + smooth_progress * (next_corner - current_corner)
        
        dx = self._params.scale[0] * position_2d[0]
        dy = self._params.scale[1] * position_2d[1]
        dz = Z_AMPLITUDE * np.sin(phase)
        
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)
    
class LissajousComplexGenerator(TrajectoryGenerator):
    """Complex Lissajous curve with 3:4 frequency ratio and phase shift."""
    _FREQ_RATIO_X = 3.0
    _FREQ_RATIO_Y = 4.0
    _PHASE_SHIFT = np.pi / 4
    
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(self._FREQ_RATIO_X * phase + self._PHASE_SHIFT)
        dy = self._params.scale[1] * np.sin(self._FREQ_RATIO_Y * phase)
        dz = Z_AMPLITUDE * np.sin(phase)
        
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)

class Figure8TrajectoryGenerator(TrajectoryGenerator):
    """Figure-8 trajectory using Lissajous curve with 1:2 frequency ratio."""
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(phase)
        dy = self._params.scale[1] * np.sin(phase / 2)
        dz = Z_AMPLITUDE * np.sin(phase)
        
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)

class LeaderRobotPublisher(Node):
    """ROS2 Node to simulate and publish the leader robot's trajectory."""
    
    def __init__(self):
        super().__init__('leader_robot_publisher')

        # ROS2 parameters
        self.publish_freq_ = DEFAULT_PUBLISH_FREQ
        self.timer_period_ = 1.0 / self.publish_freq_
        self._dt = self.timer_period_

        # Make trajectory parameters configurable via ROS2 params
        traj_type_str = self.declare_parameter('trajectory_type', TrajectoryType.FIGURE_8.value).value
        self._trajectory_type = TrajectoryType(traj_type_str)                
        self._randomize_params = self.declare_parameter('randomize_params', False).value
        self.model_path_ = self.declare_parameter('model_path', DEFAULT_MUJOCO_MODEL_PATH).value

        # Robot parameters
        self.n_joints = N_JOINTS
        self.ee_body_name = EE_BODY_NAME
        self.tcp_offset = TCP_OFFSET.copy()
        self.joint_limits_lower = JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = JOINT_LIMITS_UPPER.copy()
        
        # Warm up before starting trajectory
        self._warm_up_duration = WARM_UP_DURATION
        self._start_pos = np.zeros(3)
        
        # Load MuJoCo model for IK
        self.model = mujoco.MjModel.from_xml_path(self.model_path_)
        self.data = mujoco.MjData(self.model)

        # Initialize IK solver
        self.ik_solver = IKSolver(
            model=self.model,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper,
            jacobian_max_iter=IK_MAX_ITER,
            jacobian_step_size= IK_STEP_SIZE,
            position_tolerance=IK_TOLERANCE,
            jacobian_damping=IK_DAMPING,
            max_joint_change=IK_MAX_JOINT_CHANGE,
            continuity_gain=IK_CONTINUITY_GAIN,
        )

        # Initialize trajectory generator
        self._trajectory_time = 0.0
        self._params = TrajectoryParameters() # Start with default params
        self._generator = self._create_generator(self._trajectory_type, self._params)
        
        # State tracking
        self._last_q_desired = np.zeros(self.n_joints)
        self.joint_names_ = [f'panda_joint{i+1}' for i in range(self.n_joints)]

        # Reset simulation to initial state
        self._reset()
        
        # create ROS2 publisher and timer
        self.joint_state_pub_ = self.create_publisher(JointState, 'local_robot/joint_states', 100)
        self.timer_ = self.create_timer(self.timer_period_, self.timer_callback)

        self.get_logger().info("Local Robot Publisher node started successfully.")

    def _reset(self) -> None:
        """ Reset parameters and simulation to initial state."""

        # Randomize params if needed
        if self._randomize_params:
            self._params = self._generate_random_params()
            self._generator = self._create_generator(self._trajectory_type, self._params)
            self.get_logger().info("Trajectory parameters randomized.")

        # Reset trajectory time
        self._trajectory_time = 0.0

        # Calculate Cartesian Start Position
        trajectory_start_pos = self._generator.compute_position(0.0)
        self._start_pos = trajectory_start_pos.copy()
        
        # Solve IK for the start position immediately
        q_start, ik_success, _ = self.ik_solver.solve(
            target_pos=self._start_pos,
            q_init=INITIAL_JOINT_CONFIG.copy(), 
            body_name=self.ee_body_name,
            tcp_offset=self.tcp_offset,
            enforce_continuity=False # False because we are teleporting/resetting
        )

        if not ik_success:
            self.get_logger().warn("IK failed for initial position! Fallback to Home.")
            q_start = INITIAL_JOINT_CONFIG.copy()

        # Set internal state to this calculated start configuration
        self._last_q_desired = q_start.copy()

        # Reset MuJoCo simulation to match
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = q_start
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)
        
        self.get_logger().info(f"Simulation reset. Spawning at: {q_start}")

    def timer_callback(self) -> None:
        """
        Steps:
        1. Update trajectory time.
        2. Generate Cartesian target position.
        3. Solve IK to get desired joint positions.
        4. Publish joint states.
        """
    
        self._trajectory_time += self._dt

        # Generate Cartesian target and compute control
        if self._trajectory_time < self._warm_up_duration:
            # Hold at the start position for the warm-up duration
            cartesian_target = self._start_pos.copy()
        else:
            # After warm-up, compute position relative to the *start* of the movement
            movement_time = self._trajectory_time - self._warm_up_duration
            cartesian_target = self._generator.compute_position(movement_time)

        # Solve IK
        q_desired, ik_success, ik_error = self.ik_solver.solve(
            target_pos=cartesian_target,
            q_init=self._last_q_desired.copy(), # Was self._q_current
            body_name=self.ee_body_name,
            tcp_offset=self.tcp_offset,
            enforce_continuity=True,
        )

        if not ik_success or q_desired is None:
            self.get_logger().warn(
                f"IK failed at t={self._trajectory_time:.3f}s, error={ik_error:.6f}m. Reusing last q_desired."
            )
            q_desired = self._last_q_desired.copy()

        # Calculate desired velocity
        qd_desired = (q_desired - self._last_q_desired) / self._dt
        self._last_q_desired = q_desired.copy()
        
        # Publish the joint states message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.name = self.joint_names_
        
        # Publish the desired state from the IK solver
        msg.position = q_desired.astype(float).tolist()
        msg.velocity = qd_desired.astype(float).tolist()
        msg.effort = []
        
        self.joint_state_pub_.publish(msg)

    # Create trajectory generator
    def _create_generator(
        self,
        trajectory_type: TrajectoryType,
        params: TrajectoryParameters,
    ) -> TrajectoryGenerator:

        generators = {
            TrajectoryType.FIGURE_8: Figure8TrajectoryGenerator,
            TrajectoryType.SQUARE: SquareTrajectoryGenerator,
            TrajectoryType.LISSAJOUS_COMPLEX: LissajousComplexGenerator,
        }

        generator_class = generators.get(trajectory_type)
        if generator_class is None:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

        return generator_class(params)

    def _generate_random_params(self) -> TrajectoryParameters:
        """Generate randomized parameters within safe operational bounds."""
        return TrajectoryParameters(
            center=np.array([
                np.random.uniform(0.4, 0.5),
                np.random.uniform(-0.2, 0.2),
                0.5,
            ], dtype=np.float64),
            
            scale=np.array([
                np.random.uniform(0.1, 0.2),
                np.random.uniform(0.1, 0.3),
            ], dtype=np.float64),
            
            frequency=np.random.uniform(0.1, 0.15),
            
            initial_phase=0.0,
        )

def main(args=None):
    rclpy.init(args=args)
    leader_node = None
    
    try:
        leader_node = LeaderRobotPublisher()
        rclpy.spin(leader_node)
    except KeyboardInterrupt:
        pass # Clean exit on Ctrl+C
    except Exception as e:
        print(f"Node failed: {e}")
    finally:
        if leader_node:
            leader_node.destroy_node()
        if rclpy.ok(): # Check if context is still valid
            rclpy.shutdown()

if __name__ == '__main__':
    main()