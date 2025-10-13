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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym


class TrajectoryType(Enum):
    """Enumeration of available trajectory types."""
    FIGURE_8 = "figure_8"
    SQUARE = "square"
    LISSAJOUS_COMPLEX = "lissajous_complex" 

@dataclass(frozen=True)
class TrajectoryParams:
    """Trajectory initial parameters."""
    center: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.4, 0.0, 0.6], dtype=np.float64)
    )
    scale: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.1, 0.3], dtype=np.float64)
    )
    frequency: float = 0.1
    initial_phase: float = 0.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        assert self.center.shape == (3,), "Center must be a 3D point."
        assert self.scale.shape == (2,), "Scale must be a 2D vector."
        assert self.frequency > 0, "Frequency must be positive."

class TrajectoryGenerator:
    """Define the blueprint for each trajectory type."""
    def __init__(self, params: TrajectoryParams):
        self._params = params
    
    @property
    def params(self) -> TrajectoryParams:
        return self._params

    def compute_position(self, t_sec: float) -> NDArray[np.float64]:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def _compute_phase(self, t: float) -> float:
        """Convert time to phase angle including frequency and initial offset.
        
        Args:
            t: Time in seconds
            
        Returns:
            Phase angle in radians
        """
        return (t * self._params.frequency * 2 * np.pi + 
                self._params.initial_phase)

class SquareTrajectoryGenerator(TrajectoryGenerator):
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
    
    def compute_velocity(self, t: float) -> NDArray[np.float64]:
        """Compute velocity for smooth square trajectory.
        
        Args:
            t: Time in seconds
            
        Returns:
            3D velocity vector
        """
        phase = self._compute_phase(t)
        t_norm = (phase % (2 * np.pi)) / (2 * np.pi)
        omega = self._params.frequency * 2 * np.pi
        
        corners = np.array([
            [1, 1],    # Top-right
            [-1, 1],   # Top-left
            [-1, -1],  # Bottom-left
            [1, -1],   # Bottom-right
        ])
        
        segment = int(t_norm * 4) % 4
        segment_progress = (t_norm * 4) % 1
        
        current_corner = corners[segment]
        next_corner = corners[(segment + 1) % 4]
        
        # Derivative of smooth interpolation
        direction = next_corner - current_corner
        d_smooth_progress = 0.5 * np.pi * np.sin(segment_progress * np.pi)
        dt_norm_dt = omega / (2 * np.pi)
        velocity_2d = direction * d_smooth_progress * 4 * dt_norm_dt
        
        vx = self._params.scale[0] * velocity_2d[0]
        vy = self._params.scale[1] * velocity_2d[1]
        
        return np.array([vx, vy, 0.0], dtype=np.float64)
    
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
    
    def compute_velocity(self, t: float) -> NDArray[np.float64]:
        """Compute velocity analytically for complex Lissajous trajectory.
        
        Args:
            t: Time in seconds
            
        Returns:
            3D velocity vector
        """
        phase = self._compute_phase(t)
        omega = self._params.frequency * 2 * np.pi
        
        vx = (self._params.scale[0] * self._FREQ_RATIO_X * omega * 
              np.cos(self._FREQ_RATIO_X * phase + self._PHASE_SHIFT))
        vy = (self._params.scale[1] * self._FREQ_RATIO_Y * omega * 
              np.cos(self._FREQ_RATIO_Y * phase))
        vz = 0.0
        
        return np.array([vx, vy, vz], dtype=np.float64)

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

    def compute_velocity(self, t: float) -> NDArray[np.float64]:
        """Compute velocity analytically for figure-8 trajectory.
        
        Args:
            t: Time in seconds
            
        Returns:
            3D velocity vector
        """
        phase = self._compute_phase(t)
        omega = self._params.frequency * 2 * np.pi
        
        vx = self._params.scale[0] * omega * np.cos(phase)
        vy = self._params.scale[1] * (omega / 2) * np.cos(phase / 2)
        vz = 0.0
        
        return np.array([vx, vy, vz], dtype=np.float64)

class LocalRobotSimulator(gym.Env):
    """Gymnasium environment for trajectory-following robot simulation."""
    def __init__(self,
                 control_freq: int = 200,
                 trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
                 randomize_params: bool = False,            
    ) -> None:
        super().__init__()
        self._dt = 1.0 / control_freq
        self._randomize_params = randomize_params
        self._trajectory_time = 0.0

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -0.5, 0.0], dtype=np.float32),
            high=np.array([1.0, 0.5, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Discrete(1)
        
        # Initialize with default parameters
        self._params = TrajectoryParams()
        self._trajectory_type = trajectory_type
        self._generator = self._create_generator(trajectory_type, self._params)
        
    def _create_generator(self,
                          trajectory_type: TrajectoryType,
                          params: TrajectoryParams) -> TrajectoryGenerator:
        
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
        """Generate randomized parameters within safe operational bounds.
        
        Returns:
            New randomized parameter set
        """
        return TrajectoryParams(
            center=np.array([
                np.random.uniform(0.3, 0.5),   # X: workspace center
                np.random.uniform(-0.2, 0.2),  # Y: lateral variation
                0.6,                            # Z: fixed height
            ], dtype=np.float64),
            scale=np.array([
                np.random.uniform(0.05, 0.2),   # X scale varies
                np.random.uniform(0.1, 0.3),    # Y scale varies
            ], dtype=np.float64),
            frequency=np.random.uniform(0.08, 0.12),  # Frequency varies slightly
            initial_phase=np.random.uniform(0, 2 * np.pi),  # Random starting phase
        )
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """ Reset the environment to initial state."""
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
        
        # Reset time to trajectory start
        self._trajectory_time = 0.0
        
        # Get initial state
        initial_position = self._generator.compute_position(self._trajectory_time)
        
        info = {
            "trajectory_type": self._trajectory_type.value,
            "center": self._params.center.copy(),
            "scale_x": self._params.scale[0],
            "scale_y": self._params.scale[1],
            "frequency": self._params.frequency,
            "initial_phase": self._params.initial_phase,
        }
        
        return initial_position.astype(np.float32), info
        
    def step(
        self,
        action: Optional[Any] = None,
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        self._trajectory_time += self._dt
        
        position = self._generator.compute_position(self._trajectory_time)
        velocity = self._generator.compute_velocity(self._trajectory_time)
        
        info = {
            "time": self._trajectory_time,
            "velocity": velocity,
            "trajectory_type": self._trajectory_type.value,
        }
        
        # Basic Gymnasium return structure
        reward = 0.0
        terminated = False
        truncated = False
        
        return position.astype(np.float32), reward, terminated, truncated, info

    def get_position_at_time(self, t: float) -> NDArray[np.float64]:
        """Query position at arbitrary time point."""
        return self._generator.compute_position(t)
    
    def get_velocity_at_time(self, t: float) -> NDArray[np.float64]:
        """Query velocity at arbitrary time point."""
        return self._generator.compute_velocity(t)