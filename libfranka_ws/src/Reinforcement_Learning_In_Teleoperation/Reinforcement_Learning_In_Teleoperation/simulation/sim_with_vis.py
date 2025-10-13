# robot_control/visualization/mujoco_viewer.py
"""
Real-time MuJoCo visualization for training.
Optional - can be disabled for faster training.
"""

from __future__ import annotations
from typing import Optional
import time

import mujoco
import mujoco.viewer


class MuJoCoViewer:
    """Real-time visualization wrapper for MuJoCo simulator.
    
    Features:
    - Opens viewer window
    - Renders at specified FPS
    - Non-blocking (doesn't slow down training)
    - Can be closed/opened dynamically
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        render_fps: int = 30,
    ) -> None:
        """Initialize viewer.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            render_fps: Rendering frame rate (Hz)
        """
        self._model = model
        self._data = data
        self._render_fps = render_fps
        self._render_dt = 1.0 / render_fps
        
        # Viewer handle
        self._viewer: Optional[mujoco.viewer.Handle] = None
        self._is_open = False
        
        # Timing
        self._last_render_time = 0.0
    
    def open(self) -> None:
        """Open viewer window."""
        if not self._is_open:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self._is_open = True
            self._last_render_time = time.time()
            print("✓ Viewer opened")
    
    def close(self) -> None:
        """Close viewer window."""
        if self._is_open and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            self._is_open = False
            print("✗ Viewer closed")
    
    def render(self) -> None:
        """Render current state (rate-limited to render_fps)."""
        if not self._is_open:
            return
        
        # Rate limiting
        current_time = time.time()
        elapsed = current_time - self._last_render_time
        
        if elapsed >= self._render_dt:
            # Sync viewer with current simulation state
            if self._viewer is not None:
                self._viewer.sync()
            
            self._last_render_time = current_time
    
    @property
    def is_open(self) -> bool:
        """Check if viewer is open."""
        return self._is_open
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()