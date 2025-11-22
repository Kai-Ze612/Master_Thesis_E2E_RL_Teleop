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
        """Compute 3D position at time t."""
        pass
    
    def _compute_phase(self, t: float) -> float:
        """Compute phase angle at time t."""
        return (t * self._params.frequency * 2 * np.pi + 
                self._params.initial_phase)

class SquareTrajectoryGenerator(TrajectoryGenerator): # import the custom interface
    """Square trajectory in XY plane with smooth corners."""
    def compute_position(self, t: float) -> NDArray[np.float64]:
        
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
        
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(self._FREQ_RATIO_X * phase + self._PHASE_SHIFT)
        dy = self._params.scale[1] * np.sin(self._FREQ_RATIO_Y * phase)
        return self._params.center + np.array([dx, dy, 0.0], dtype=np.float64)

class Figure8TrajectoryGenerator(TrajectoryGenerator):
    """Figure-8 trajectory using Lissajous curve with 1:2 frequency ratio."""
    def compute_position(self, t: float) -> NDArray[np.float64]:
        
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
                0.5,
            ], dtype=np.float64),
            
            scale=np.array([
                np.random.uniform(0.1, 0.2),
                np.random.uniform(0.1, 0.2),
            ], dtype=np.float64),

            frequency=np.random.uniform(0.05, 0.15),

            initial_phase=0.0,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment and initialize joint configuration."""

        super().reset(seed=seed)

        # Apply parameter randomization if enabled
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

        # Reset trajectory time
        self._trajectory_time = 0.0
        self._tick = 0
        
        # Calculate trajectory start position
        self._traj_start_pos = self._generator.compute_position(0.0)
        
        # Calculate Home Position
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = INITIAL_JOINT_CONFIG
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)
        self._home_pos_cartesian = self.get_cartesian_position() # Helper function exists    

        self._q_current = INITIAL_JOINT_CONFIG.copy()
        self._qd_current = np.zeros(self.n_joints)
        self._last_q_desired = INITIAL_JOINT_CONFIG.copy()
        
        # Info dictionary
        info = {
            "trajectory_type": self._trajectory_type.value,
            "joint_pos": self._q_current.copy(),
            "center": self._params.center.copy(),
        }

        return self._q_current.astype(np.float32), info
    
    def _normalize_angle(self, angle: np.ndarray) -> np.ndarray:
        """Normalize an angle or array of angles to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _get_inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a_desired: np.ndarray) -> np.ndarray:
        """
        Compute Torque using MuJoCo Inverse Dynamics.
        Equation: tau = M(q)*a_desired + C(q,v)*v + G(q)
        """
        # 1. Update MuJoCo with the CURRENT Robot State
        self.data.qpos[:self.n_joints] = q
        self.data.qvel[:self.n_joints] = v
        
        # 2. Set the DESIRED Acceleration (from PD)
        self.data.qacc[:self.n_joints] = a_desired

        # 3. Compute Inverse Dynamics
        # This calculates the forces required to produce 'a_desired'
        mujoco.mj_inverse(self.model, self.data)

        return self.data.qfrc_inverse[:self.n_joints].copy()
    
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
            alpha = self._trajectory_time / self._warm_up_duration
            # Cubic easing for smoother start/stop (optional, but better than linear)
            alpha_smooth = alpha * alpha * (3 - 2 * alpha) 
            cartesian_target = (1 - alpha_smooth) * self._home_pos_cartesian + alpha_smooth * self._traj_start_pos
            
        else:
            # Normal Trajectory Generation
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

        # 1. Get Current State
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()

        # 2. Calculate Error
        q_error = self._normalize_angle(q_desired - q_current)
        qd_error = qd_desired - qd_current

        # 3. PD Control -> Output is DESIRED ACCELERATION (not Torque)
        # acc = Kp * (pos_err) + Kd * (vel_err)
        acc_desired = self.kp_local * q_error + self.kd_local * qd_error

        # 4. Inverse Dynamics -> Output is REQUIRED TORQUE
        # tau = M(q)*acc + C + G
        tau_control = self._get_inverse_dynamics(q_current, qd_current, acc_desired)
        
        # 5. Apply to Simulation
        self.data.ctrl[:self.n_joints] = tau_control

        # 6. Step Physics
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