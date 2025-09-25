"""
This script serves as a local trajectory generator for the leader robot.

Current trajectory types:
- Figure-8: Smooth continuous motion, good for generalization
- Square: Sharp turns with smooth corners, tests abrupt changes
- Star: Complex pattern with configurable points, tests precision

For simplicity, all trajectories are confined to the XY plane with a fixed Z height.
"""

import numpy as np
import gymnasium as gym
from enum import Enum
from typing import Dict, Any, Optional, Tuple

class TrajectoryType(Enum):
    """Enumeration of available trajectory types."""
    FIGURE_8 = "figure_8"
    SQUARE = "square"
    STAR = "star"

class LocalRobotSimulator(gym.Env):
    def __init__(self, 
                 control_freq: int = 200,
                 trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
                 randomize_params: bool = False):
        super().__init__()
        
        # Basic configurations
        self.dt = 1.0 / control_freq
        self.trajectory_type = trajectory_type
        self.randomize_params = randomize_params
        self.traj_time = 0.0
        
        # Default trajectory parameters (can be overridden)
        self.default_params = {
            'center': np.array([0.4, 0.0, 0.6]),  # Fixed center position
            'scale': np.array([0.1, 0.3]),        # Only X and Y scaling
            'frequency': 0.1,                     # Fixed frequency
            'initial_phase': 0.0,
            'n_segments': 4,  # For square and star trajectories
        }
        
        # Current trajectory parameters
        self.traj_params = self.default_params.copy()
        
    def set_trajectory_params(self, **params):
        """Update trajectory parameters."""
        for key, value in params.items():
            if key in self.traj_params:
                self.traj_params[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")
    
    def _generate_random_params(self) -> Dict[str, Any]:
        """Generate randomized trajectory parameters - scale and center vary."""
        return {
            'center': np.array([
                np.random.uniform(0.3, 0.5),      # X center varies within workspace
                np.random.uniform(-0.15, 0.15),   # Y center varies around middle
                0.6                               # Z height remains fixed
            ]),
            'scale': np.array([
                np.random.uniform(0.1, 0.2),      # X scale varies
                np.random.uniform(0.1, 0.2),      # Y scale varies
            ]),
            'frequency': 0.1,                      # Fixed frequency
            'initial_phase': 0.0,                  # Fixed initial phase
            'n_segments': 4,                       # Fixed segments
        }
    
    def get_position_at_time(self, t_sec: float) -> np.ndarray:
        """Generate position at given time based on current trajectory type."""
        if self.trajectory_type == TrajectoryType.FIGURE_8:
            return self._figure_8_position(t_sec)
        elif self.trajectory_type == TrajectoryType.SQUARE:
            return self._square_position(t_sec)
        elif self.trajectory_type == TrajectoryType.STAR:
            return self._star_position(t_sec)
        else:
            raise ValueError(f"Unknown trajectory type: {self.trajectory_type}")
    
    def _figure_8_position(self, t_sec: float) -> np.ndarray:
        """Figure-8 trajectory in XY plane only."""
        t = t_sec * self.traj_params['frequency'] * 2 * np.pi + self.traj_params['initial_phase']
        dx = self.traj_params['scale'][0] * np.sin(t)
        dy = self.traj_params['scale'][1] * np.sin(t / 2)
        return self.traj_params['center'] + np.array([dx, dy, 0.0])
    
    def _square_position(self, t_sec: float) -> np.ndarray:
        """Square trajectory in XY plane with smooth corners."""
        t = t_sec * self.traj_params['frequency'] * 2 * np.pi + self.traj_params['initial_phase']
        t_norm = (t % (2 * np.pi)) / (2 * np.pi)  # Normalize to [0, 1]
        
        # Define square corners
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
        
        dx = self.traj_params['scale'][0] * position_2d[0]
        dy = self.traj_params['scale'][1] * position_2d[1]
        return self.traj_params['center'] + np.array([dx, dy, 0.0])
    
    def _star_position(self, t_sec: float) -> np.ndarray:
        """Star trajectory in XY plane with configurable number of points."""
        t = t_sec * self.traj_params['frequency'] * 2 * np.pi + self.traj_params['initial_phase']
        n_points = self.traj_params['n_segments']
        
        # Create star shape by modulating the radius
        base_radius = 1.0
        radius_variation = 0.5 * np.sin(n_points * t)
        radius = base_radius + radius_variation
        
        dx = self.traj_params['scale'][0] * radius * np.cos(t)
        dy = self.traj_params['scale'][1] * radius * np.sin(t)
        return self.traj_params['center'] + np.array([dx, dy, 0.0])
    
    def get_velocity_at_time(self, t_sec: float) -> np.ndarray:
        """Calculate velocity by numerical differentiation."""
        epsilon = 1e-6
        pos_current = self.get_position_at_time(t_sec)
        pos_next = self.get_position_at_time(t_sec + epsilon)
        return (pos_next - pos_current) / epsilon
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset trajectory to start position."""
        super().reset(seed=seed)
        
        # Update parameters if randomization is enabled
        if self.randomize_params:
            self.traj_params.update(self._generate_random_params())
        
        # Apply any options provided
        if options:
            for key, value in options.items():
                if key == 'trajectory_type':
                    self.trajectory_type = TrajectoryType(value)
                elif key in self.traj_params:
                    self.traj_params[key] = value
        
        # Reset time
        self.traj_time = 0.0
        
        # Get initial position
        start_pos = self.get_position_at_time(self.traj_time)
        
        # Prepare info dictionary
        info = {
            "trajectory_type": self.trajectory_type.value,
            "center": self.traj_params['center'].copy(),
            "scale_x": self.traj_params['scale'][0],
            "scale_y": self.traj_params['scale'][1],
            "frequency": self.traj_params['frequency'],
            "initial_phase": self.traj_params['initial_phase'],
        }
        
        return start_pos, info
    
    def step(self, action: Optional[Any] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advance one time step in the trajectory."""
        self.traj_time += self.dt
        
        position = self.get_position_at_time(self.traj_time)
        velocity = self.get_velocity_at_time(self.traj_time)
        
        # Basic reward (can be customized based on specific requirements)
        reward = 0.0
        terminated = False
        truncated = False
        
        info = {
            "time": self.traj_time,
            "velocity": velocity,
            "trajectory_type": self.trajectory_type.value
        }
        
        return position, reward, terminated, truncated, info
    
    def change_trajectory(self, new_trajectory: TrajectoryType, **params):
        """Change trajectory type and optionally update parameters."""
        self.trajectory_type = new_trajectory
        if params:
            self.set_trajectory_params(**params)