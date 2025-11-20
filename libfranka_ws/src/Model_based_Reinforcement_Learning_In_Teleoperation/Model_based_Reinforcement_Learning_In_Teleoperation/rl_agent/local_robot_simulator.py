"""
In the experiment, we assume the leader robot is a perfect trajectory generator.

In this script, we implement a local robot simulator that generates various 2D trajectories. (all trajectories are in the XY plane with a fixed Z height for simplicity)

Current trajectory types:
- Figure-8: Smooth continuous motion, good for generalization
- Square: Sharp turns with smooth corners, tests abrupt changes
- Lissajous_Complex: A higher-order, complex, smooth trajectory, tests precision and complex tracking

The trajectory parameters can be randomized within reasonable bounds to enhance robustness during training.
"""

# Python imports
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
import mujoco

from Model_based_Reinforcement_Learning_In_Teleoperation.utils.inverse_kinematics import IKSolver
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    DEFAULT_MUJOCO_MODEL_PATH,
    N_JOINTS,
    EE_BODY_NAME,
    TCP_OFFSET,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    INITIAL_JOINT_CONFIG,
    KP_LOCAL,
    KD_LOCAL,
    DEFAULT_CONTROL_FREQ,
    IK_MAX_ITER,
    IK_TOLERANCE,
    IK_DAMPING,
    IK_MAX_JOINT_CHANGE,
    IK_CONTINUITY_GAIN,
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

@dataclass(frozen=True) # Make immutable
class TrajectoryParams:
    """Trajectory initial parameters."""
    center: NDArray[np.float64] = field(
        default_factory=lambda: TRAJECTORY_CENTER.copy()
    ) # Make center mutable and get it a default value by using default_factory
    
    scale: NDArray[np.float64] = field(
        default_factory=lambda: TRAJECTORY_SCALE.copy()
    )
    
    frequency: float = TRAJECTORY_FREQUENCY
    initial_phase: float = 0.0

    # This will help check the validity of parameters after initialization
    def __post_init__(self) -> None:
        """Validate parameters."""
        assert self.center.shape == (3,), "Center must be a 3D point."
        assert self.scale.shape == (2,), "Scale must be a 2D vector."
        assert self.frequency > 0, "Frequency must be positive."

class TrajectoryGenerator(ABC): # Define the interface
    """This class is only position computation for the remote robot"""

    def __init__(self,
                 params: TrajectoryParams):
        self._params = params
        
    @property
    def params(self) -> TrajectoryParams:
        return self._params

    @abstractmethod
    def compute_position(self, t: float) -> NDArray[np.float64]:
        """Compute position at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            3D position vector
        """
        pass
    
    def _compute_phase(self, t: float) -> float:
        """Compute phase angle at time t."""
        return (t * self._params.frequency * 2 * np.pi + 
                self._params.initial_phase)

class SquareTrajectoryGenerator(TrajectoryGenerator): # import the custom interface
    """Square trajectory in XY plane with smooth corners."""
    def compute_position(self, t: float) -> NDArray[np.float64]:
        """Generate square pattern in XY plane.
        
        Args:
            t: Time in seconds
            
        Returns:
            3D position with Z fixed at center height
        """
        phase = self._compute_phase(t)
        t_norm = (phase % (2 * np.pi)) / (2 * np.pi)  # Normalize to [0, 1]
        
        # Define square corners (fixed, no need for n_segments)
        corners = np.array([
            [1, 1],    # Top-right
            [-1, 1],   # Top-left
            [-1, -1],  # Bottom-left
            [1, -1],   # Bottom-right
        ])
        
        # Determine which segment we're in
        segment = int(t_norm * 4) % 4
        segment_progress = (t_norm * 4) % 1
        
        # Linear interpolation between corners
        current_corner = corners[segment]
        next_corner = corners[(segment + 1) % 4]
        
        # Smooth interpolation using cosine
        smooth_progress = 0.5 * (1 - np.cos(segment_progress * np.pi))
        position_2d = current_corner + smooth_progress * (next_corner - current_corner)
        
        dx = self._params.scale[0] * position_2d[0]
        dy = self._params.scale[1] * position_2d[1]
        
        return self._params.center + np.array([dx, dy, 0.0], dtype=np.float64)

class LissajousComplexGenerator(TrajectoryGenerator):
    """Complex Lissajous curve with 3:4 frequency ratio and phase shift."""
    
    _FREQ_RATIO_X = 3.0
    _FREQ_RATIO_Y = 4.0
    _PHASE_SHIFT = np.pi / 4
    
    def compute_position(self, t: float) -> NDArray[np.float64]:
        """Generate complex Lissajous pattern for precision testing.
        
        Mathematical form: x(t) = A*sin(3ωt + π/4), y(t) = B*sin(4ωt)
        
        Args:
            t: Time in seconds
            
        Returns:
            3D position with Z fixed at center height
        """
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(self._FREQ_RATIO_X * phase + self._PHASE_SHIFT)
        dy = self._params.scale[1] * np.sin(self._FREQ_RATIO_Y * phase)
        return self._params.center + np.array([dx, dy, 0.0], dtype=np.float64)

class Figure8TrajectoryGenerator(TrajectoryGenerator):
    """Figure-8 trajectory using Lissajous curve with 1:2 frequency ratio."""
    def compute_position(self, t: float) -> NDArray[np.float64]:
        """Generate smooth figure-8 pattern in XY plane.
        
        Mathematical form: x(t) = A*sin(ωt), y(t) = B*sin(ωt/2)
        
        Args:
            t: Time in seconds
            
        Returns:
            3D position with Z fixed at center height
        """
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(phase)
        dy = self._params.scale[1] * np.sin(phase / 2)
        return self._params.center + np.array([dx, dy, 0.0], dtype=np.float64)

class LocalRobotSimulator(gym.Env):
    """The main trajectory generator class"""
    def __init__(
        self,
        model_path: str = DEFAULT_MUJOCO_MODEL_PATH,
        control_freq: int = DEFAULT_CONTROL_FREQ,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_params: bool = False,
        # Franka Panda Parameters
        joint_limits_lower: Optional[NDArray[np.float64]] = JOINT_LIMITS_LOWER,
        joint_limits_upper: Optional[NDArray[np.float64]] = JOINT_LIMITS_UPPER,
        # IK solver parameters
        ik_max_iter: int = IK_MAX_ITER,
        ik_tolerance: float = IK_TOLERANCE,
        ik_damping: float = IK_DAMPING,
        # PD gains for local robot
        kp_local: NDArray[np.float64] = KP_LOCAL,
        kd_local: NDArray[np.float64] = KD_LOCAL,
        # Warm up before start
        warm_up_duration: float = WARM_UP_DURATION,
    ) -> None:
        super().__init__()
        
        self.n_joints = N_JOINTS
        self.ee_body_name = EE_BODY_NAME
        self.tcp_offset = TCP_OFFSET.copy()
        self._dt = 1.0 / control_freq
        self._control_freq = control_freq
        self._randomize_params = randomize_params

        self._tick = 0  # We use tick to count the steps
        
        # Warm-up time before starting trajectory
        self._warm_up_duration = warm_up_duration
        self._start_pos = np.zeros(3)
        
        # PD gains for local robot
        self.kd_local = kd_local.copy()
        self.kp_local = kp_local.copy()

        # Joint limits
        self.joint_limits_lower = joint_limits_lower.copy()
        self.joint_limits_upper = joint_limits_upper.copy()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Simulation frequency and substeps
        sim_freq = int(1.0 / self.model.opt.timestep)
        if sim_freq % control_freq != 0:
            raise ValueError(
                f"Simulation frequency ({sim_freq} Hz) must be a multiple of control frequency ({control_freq} Hz)."
            )
        self.n_substeps = sim_freq // control_freq

        # Initialize IK solver
        self.ik_solver = IKSolver(
            model=self.model,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper,
            jacobian_max_iter=ik_max_iter,
            position_tolerance=ik_tolerance,
            jacobian_damping=ik_damping,
            max_joint_change=IK_MAX_JOINT_CHANGE,
            continuity_gain=IK_CONTINUITY_GAIN,
        )

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=self.joint_limits_lower.astype(np.float32),
            high=self.joint_limits_upper.astype(np.float32),
            shape=(self.n_joints,),
            dtype=np.float32
        )

        # Dummy action space
        self.action_space = gym.spaces.Discrete(1)

        # Initialize trajectory generator
        self._trajectory_time = 0.0
        self._params = TrajectoryParams()
        self._trajectory_type = trajectory_type
        self._generator = self._create_generator(trajectory_type, self._params)

        # State tracking
        self._q_current = np.zeros(self.n_joints)
        self._qd_current = np.zeros(self.n_joints)
        self._last_q_desired = np.zeros(self.n_joints)

    def _create_generator(
        self,
        trajectory_type: TrajectoryType,
        params: TrajectoryParams
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

    def _generate_random_params(self) -> TrajectoryParams:
        """Generate randomized parameters within safe operational bounds."""
        return TrajectoryParams(
            center=np.array([
                np.random.uniform(0.4, 0.5),
                np.random.uniform(-0.2, 0.2),
                0.6,
            ], dtype=np.float64),
            
            scale=np.array([
                np.random.uniform(0.1, 0.2),
                np.random.uniform(0.1, 0.3),
            ], dtype=np.float64),

            frequency=np.random.uniform(0.1, 0.2),

            initial_phase=0.0,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment and initialize joint configuration."""

        super().reset(seed=seed)

        # 1. Apply parameter randomization if enabled
        if self._randomize_params:
            self._params = self._generate_random_params()
            self._generator = self._create_generator(self._trajectory_type, self._params)

        # Apply optional overrides
        if options:
            if 'trajectory_type' in options:
                new_type = TrajectoryType(options['trajectory_type'])
                if new_type != self._trajectory_type:
                    self._trajectory_type = new_type
                    self._generator = self._create_generator(new_type, self._params)

        # 2. Reset trajectory time
        self._trajectory_time = 0.0
        self._tick = 0
        
        # 3. Calculate where the trajectory mathematically starts
        trajectory_start_pos = self._generator.compute_position(0.0)
        self._start_pos = trajectory_start_pos.copy()

        # 4. [CRITICAL FIX] Solve IK for this start position immediately
        # We use INITIAL_JOINT_CONFIG only as a seed for the IK solver
        q_start, ik_success, _ = self.ik_solver.solve(
            target_pos=trajectory_start_pos,
            q_init=INITIAL_JOINT_CONFIG, # Use home as guess
            body_name=self.ee_body_name,
            tcp_offset=self.tcp_offset,
            enforce_continuity=False # False because this is a teleport reset
        )

        if not ik_success:
            print(f"[WARNING] Could not solve IK for trajectory start: {trajectory_start_pos}")
            # Fallback to home if IK fails (though it shouldn't if params are safe)
            q_start = INITIAL_JOINT_CONFIG.copy()

        # 5. Initialize joint state targets to this new start position
        self._q_current = q_start.copy()
        self._qd_current = np.zeros(self.n_joints)
        self._last_q_desired = q_start.copy() # <--- Prevents velocity spike on Step 1

        # 6. Reset MuJoCo simulation to this specific q_start
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = q_start
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)

        # Info dictionary
        info = {
            "trajectory_type": self._trajectory_type.value,
            "joint_pos": self._q_current.copy(),
            "trajectory_start_pos": trajectory_start_pos,
            "center": self._params.center.copy(),
            "scale_x": self._params.scale[0],
            "scale_y": self._params.scale[1],
            "frequency": self._params.frequency,
        }

        return self._q_current.astype(np.float32), info

    def step(
        self,
        action: Optional[Any] = None,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Generate next joint configuration with realistic dynamics.

        Pipeline:
            1. Advance time
            2. Generate Cartesian position
            3. Solve IK → q_desired
            4. PD control → τ_control
            5. MuJoCo simulation
            6. Read actual joint state
        """

        self._trajectory_time += self._dt
        self._tick += 1
        
        
        # Add wait logic
        if self._trajectory_time < self._warm_up_duration:
            # Hold at the start position for the wait duration
            cartesian_target = self._start_pos.copy()
        else:
            # After wait, compute position relative to the *start* of the movement
            # This ensures the trajectory starts from t=0 *after* the wait
            movement_time = self._trajectory_time - self._warm_up_duration
            cartesian_target = self._generator.compute_position(movement_time)

        # IK Solver
        q_desired, ik_success, ik_error = self.ik_solver.solve(
            target_pos=cartesian_target,
            q_init=self._q_current,
            body_name=self.ee_body_name,
            tcp_offset=self.tcp_offset,
            enforce_continuity=True,
        )

        if not ik_success or q_desired is None:
            print(f"IK failed at t={self._trajectory_time:.3f}s, error={ik_error:.6f}m")
            q_desired = self._last_q_desired.copy()

        # Desired joint velocities
        qd_desired = (q_desired - self._last_q_desired) / self._dt
        self._last_q_desired = q_desired.copy()

        # PD controller for local robot tracking
        q_error = q_desired - self.data.qpos[:self.n_joints]
        qd_error = qd_desired - self.data.qvel[:self.n_joints]

        tau_control = self.kp_local * q_error + self.kd_local * qd_error
        self.data.ctrl[:self.n_joints] = tau_control

        # Simulate MuJoCo dynamics
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Read actual joint state from simulation
        self._q_current = self.data.qpos[:self.n_joints].copy()
        self._qd_current = self.data.qvel[:self.n_joints].copy()

        # Compute tracking error
        cartesian_achieved = self.get_cartesian_position()
        tracking_error = np.linalg.norm(cartesian_target - cartesian_achieved)

        # Info dictionary
        info = {
            "time": self._trajectory_time,
            "trajectory_type": self._trajectory_type.value,
            "cartesian_target": cartesian_target,
            "cartesian_achieved": cartesian_achieved,
            "tracking_error": tracking_error,
            "joint_pos": self._q_current.copy(),
            "joint_vel": self._qd_current.copy(),
            "ik_success": ik_success,
            "ik_error": ik_error,
        }

        # Basic Gymnasium return structure
        reward = 0.0
        terminated = False
        truncated = False

        return self._q_current.astype(np.float32), self._qd_current.astype(np.float32), reward, terminated, truncated, info

    def get_joint_state(self) -> dict:
        """Get current joint positions and velocities."""
        return {
            "joint_pos": self._q_current.copy(),
            "joint_vel": self._qd_current.copy(),
        }

    def get_cartesian_position(self) -> NDArray[np.float64]:
        """Get current end-effector Cartesian position."""
        ee_id = self.model.body(self.ee_body_name).id
        flange_pos = self.data.xpos[ee_id].copy()
        flange_rot = self.data.xmat[ee_id].reshape(3, 3)
        return flange_pos + flange_rot @ self.tcp_offset

    def get_position_at_time(self, t: float) -> NDArray[np.float64]:
        """Query Cartesian position at arbitrary time point."""
        if t < self._warm_up_duration:
            return self._start_pos.copy()
        else:
            movement_time = t - self._warm_up_duration
            return self._generator.compute_position(movement_time)
    
    def get_current_tick(self) -> int:
        """Get the current simulation tick."""
        return self._tick