"""
This script serves as a local trajectory generator for the leader robot.

Current trajectory types:
- Figure-8: Smooth continuous motion, good for generalization.
"""

import numpy as np
import gymnasium as gym

class LocalRobotSimulator(gym.Env):
    def __init__(self, control_freq: int):
        super().__init__()  # Initialize the parent Env class
        
        # --- Fixed Trajectory Parameters ---
        self.dt = 1.0 / control_freq
        
        # Completely fixed parameters - no randomization
        self.traj_center = np.array([0.4, 0.0, 0.6])
        self.traj_scale = np.array([0.1, 0.3, 0.0])
        self.traj_freq = 0.1
        self.initial_phase = 0.0
        
        # Time tracking
        self.traj_time = 0.0
        
    def get_position_at_time(self, t_sec: float) -> np.ndarray:
        """Generate figure-8 position at given time."""
        t = t_sec * self.traj_freq * 2 * np.pi + self.initial_phase
        dx = self.traj_scale[0] * np.sin(t)
        dy = self.traj_scale[1] * np.sin(t / 2)
        return self.traj_center + np.array([dx, dy, 0])
    
    def reset(self, seed=None, options=None):
        """Reset to start of trajectory - completely deterministic."""
        super().reset(seed=seed)
        self.traj_time = 0.0  # Always start at time 0
        
        start_pos = self.get_position_at_time(self.traj_time)
        
        info = {
            "scale_x": self.traj_scale[0],
            "scale_y": self.traj_scale[1],
            "center": self.traj_center.copy(),
            "frequency": self.traj_freq
        }
        return start_pos, info
    
    def step(self, action=None):
        """Advance one time step in the trajectory."""
        self.traj_time += self.dt
        
        position = self.get_position_at_time(self.traj_time)
        reward = 0.0
        terminated = False
        truncated = False
        info = {"time": self.traj_time}
        
        return position, reward, terminated, truncated, info