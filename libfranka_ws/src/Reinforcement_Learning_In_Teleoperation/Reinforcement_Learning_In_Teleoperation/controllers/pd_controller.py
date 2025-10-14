"""
Adaptive PD controller for teleoperation with variable delay.

Adjusts PD gains linearly based on communication delay:
- High delay → Lower gains (more conservative, avoid instability)
- Low delay → Higher gains (more aggressive, better tracking)

Adaptation law: k(τ) = k_nominal * max(min_ratio, 1 - τ/τ_threshold)

desired acc: = k_p(τ) * (q_target - q_current) + k_d(τ) * (qd_target - qd_current)
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from numpy.typing import NDArray


class AdaptivePDController:
    
    def __init__(
        self,
        n_joints: int,
        kp_nominal: Optional[NDArray[np.float64]] = None,
        kd_nominal: Optional[NDArray[np.float64]] = None,
        min_gain_ratio: float = 0.1,
        decay_rate: float = 15.0,
    ):

        self.n_joints = n_joints

        # Adaptive PD gains parameters
        self.min_gain_ratio = np.clip(min_gain_ratio, 0.1, 1.0)
        self.decay_rate = max(decay_rate, 1e-3)

        # Set Nominal Proportional Gains (Kp)
        if kp_nominal is not None:
            if kp_nominal.shape != (n_joints,):
                raise ValueError(f"kp_nominal must have shape ({n_joints},), got {kp_nominal.shape}")
            self.kp_nominal = kp_nominal.copy()
        else:
            # Default gains tuned for a Franka Panda robot at 500 Hz
            if n_joints == 7:
                self.kp_nominal = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
            else:
                self.kp_nominal = 500.0 * np.ones(n_joints)

        # Set Nominal Derivative Gains (Kd)
        if kd_nominal is not None:
            if kd_nominal.shape != (n_joints,):
                raise ValueError(f"kd_nominal must have shape ({n_joints},), got {kd_nominal.shape}")
            self.kd_nominal = kd_nominal.copy()
        else:
            # Default to critical damping if not provided
            self.kd_nominal = 2.0 * np.sqrt(self.kp_nominal)

        # Initialize Current State
        self.kp_current = self.kp_nominal.copy()
        self.kd_current = self.kd_nominal.copy()
        self.current_delay = 0.0
        self.current_gain_ratio = 1.0
    
    def _compute_gain_ratio(self, delay: float) -> float:
        """Computes the gain ratio using an exponential decay function."""
        if delay <= 0:
            return 1.0

        # Formula ensures the ratio decays from 1.0 towards min_gain_ratio
        ratio = (1.0 - self.min_gain_ratio) * np.exp(-self.decay_rate * delay) + self.min_gain_ratio
        
        # Clip for absolute safety, though the formula should not exceed bounds.
        return np.clip(ratio, self.min_gain_ratio, 1.0)

    def update_gains(self, delay: float) -> None:
        """
        Updates the current Kp and Kd gains based on the measured delay.

        Args:
            delay: The measured communication delay in seconds.
        """
        self.current_delay = delay
        self.current_gain_ratio = self._compute_gain_ratio(delay)

        # Scale nominal gains to get the current gains
        self.kp_current = self.kp_nominal * self.current_gain_ratio
        self.kd_current = self.kd_nominal * self.current_gain_ratio

    def compute_desired_acceleration(
        self,
        q_current: NDArray[np.float64],
        qd_current: NDArray[np.float64],
        q_target: NDArray[np.float64],
        qd_target: Optional[NDArray[np.float64]] = None,
        delay: Optional[float] = None,
    ) -> NDArray[np.float64]:
        
        # Validate inputs
        if q_current.shape != (self.n_joints,) or qd_current.shape != (self.n_joints,) or q_target.shape != (self.n_joints,):
            raise ValueError(f"Input arrays must have shape ({self.n_joints},).")
        
        if qd_target is None:
            qd_target = np.zeros(self.n_joints)
        elif qd_target.shape != (self.n_joints,):
            raise ValueError(f"qd_target must have shape ({self.n_joints},).")

        # Update gains if a new delay measurement is provided
        if delay is not None:
            self.update_gains(delay)

        # Compute position and velocity errors
        e_pos = q_target - q_current
        e_vel = qd_target - qd_current

        # Apply the adaptive PD control law
        qdd_desired = self.kp_current * e_pos + self.kd_current * e_vel

        return qdd_desired

    def get_current_gains(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns the current adapted Kp and Kd gains."""
        return self.kp_current.copy(), self.kd_current.copy()