"""
A generalized PD Controller for different joint number robots

For research purpose, we record the history data
"""

import numpy as np
import time
from typing import Optional, Tuple, Dict, List, Union


class PDController:
    """
    Enhanced PD Controller for robotic applications with research capabilities.
    
    Implements the PD control law:
    q̈_desired = Kp * position_error + Kd * velocity_error
    
    For use with inverse dynamics: τ = M(q)q̈_desired + C(q,q̇)q̇ + G(q)
    """
    
    def __init__(self, 
                 kp: Union[List[float], np.ndarray], 
                 kd: Union[List[float], np.ndarray], 
                 torque_limits: Union[List[float], np.ndarray],
                 joint_limits_lower: Optional[Union[List[float], np.ndarray]] = None,
                 joint_limits_upper: Optional[Union[List[float], np.ndarray]] = None,
                 enable_history: bool = True,
                 max_history_size: int = 1000):
        
        # Convert inputs to numpy arrays
        self.kp = np.array(kp, dtype=np.float64)
        self.kd = np.array(kd, dtype=np.float64)
        self.torque_limits = np.array(torque_limits, dtype=np.float64)
        
        # Get number of joints
        self.num_joints = len(self.kp)
        
        # Input validation
        self._validate_initialization_inputs()
        
        # Joint limits (optional)
        self.joint_limits_lower = np.array(joint_limits_lower, dtype=np.float64) if joint_limits_lower is not None else None
        self.joint_limits_upper = np.array(joint_limits_upper, dtype=np.float64) if joint_limits_upper is not None else None
        
        # Validate joint limits if provided
        self._validate_joint_limits()
        
        # Research and tracking capabilities
        self.enable_history = enable_history
        self.max_history_size = max_history_size
        self.control_history: List[Dict] = []
        self.adaptation_history: List[Dict] = []
        
        # Performance statistics
        self.total_control_calls = 0
        self.last_control_time = None

        # Gain bounds for safety (can be updated for research)
        self.kp_min = 0.01 * self.kp
        self.kp_max = 10.0 * self.kp
        self.kd_min = 0.01 * self.kd
        self.kd_max = 10.0 * self.kd
        
    def _validate_initialization_inputs(self) -> None:
        """Validate initialization parameters."""
        # Check dimensions match
        if not (len(self.kp) == len(self.kd) == len(self.torque_limits)):
            raise ValueError(f"Dimension mismatch: Kp={len(self.kp)}, Kd={len(self.kd)}, torque_limits={len(self.torque_limits)}")
        
        # Check for valid gains
        if np.any(self.kp < 0):
            raise ValueError("All Kp gains must be non-negative")
        if np.any(self.kd < 0):
            raise ValueError("All Kd gains must be non-negative")
        
        # Check for valid torque limits
        if np.any(self.torque_limits <= 0):
            raise ValueError("All torque limits must be positive")
        
        # Warn about zero gains
        if np.any(self.kp == 0):
            print("Warning: Some Kp gains are zero - this may result in poor tracking")
        if np.any(self.kd == 0):
            print("Warning: Some Kd gains are zero - this may result in oscillations")
    
    def _validate_joint_limits(self) -> None:
        """Validate joint limits if provided."""
        if self.joint_limits_lower is not None:
            if len(self.joint_limits_lower) != self.num_joints:
                raise ValueError(f"joint_limits_lower must have {self.num_joints} elements")
        
        if self.joint_limits_upper is not None:
            if len(self.joint_limits_upper) != self.num_joints:
                raise ValueError(f"joint_limits_upper must have {self.num_joints} elements")
        
        if self.joint_limits_lower is not None and self.joint_limits_upper is not None:
            if np.any(self.joint_limits_lower >= self.joint_limits_upper):
                raise ValueError("All lower joint limits must be less than upper limits")
    
    def compute_desired_acceleration(self, 
                                   target_positions: Union[List[float], np.ndarray],
                                   target_velocities: Union[List[float], np.ndarray],
                                   current_positions: Union[List[float], np.ndarray],
                                   current_velocities: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute desired joint accelerations using PD control law.
        
        Args:
            target_positions: Desired joint positions (rad)
            target_velocities: Desired joint velocities (rad/s)
            current_positions: Current joint positions (rad)
            current_velocities: Current joint velocities (rad/s)
            
        Returns:
            Desired joint accelerations (rad/s²)
            
        Raises:
            ValueError: If input dimensions don't match expected joints
        """
        # Convert to numpy arrays
        target_pos = np.array(target_positions, dtype=np.float64)
        target_vel = np.array(target_velocities, dtype=np.float64)
        current_pos = np.array(current_positions, dtype=np.float64)
        current_vel = np.array(current_velocities, dtype=np.float64)
        
        # Validate input dimensions
        self._validate_control_inputs(target_pos, target_vel, current_pos, current_vel)
        
        # Check joint limits if provided
        self._check_joint_limits(current_pos, target_pos)
        
        # Compute control errors
        position_error = target_pos - current_pos
        velocity_error = target_vel - current_vel
        
        # Apply PD control law
        desired_accelerations = self.kp * position_error + self.kd * velocity_error
        
        # Compute control effort for monitoring
        control_effort = np.linalg.norm(desired_accelerations)
        
        # Store control history for research
        if self.enable_history:
            self._store_control_history(position_error, velocity_error, desired_accelerations, control_effort)
        
        # Update statistics
        self.total_control_calls += 1
        self.last_control_time = time.time()
        
        return desired_accelerations
    
    def _validate_control_inputs(self, target_pos: np.ndarray, target_vel: np.ndarray,
                               current_pos: np.ndarray, current_vel: np.ndarray) -> None:
        """Validate control input dimensions."""
        expected_shape = (self.num_joints,)
        
        inputs = [
            ("target_positions", target_pos),
            ("target_velocities", target_vel),
            ("current_positions", current_pos),
            ("current_velocities", current_vel)
        ]
        
        for name, arr in inputs:
            if arr.shape != expected_shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {expected_shape}")
    
    def _check_joint_limits(self, current_pos: np.ndarray, target_pos: np.ndarray) -> None:
        """Check if positions are within joint limits."""
        if self.joint_limits_lower is not None:
            if np.any(current_pos < self.joint_limits_lower):
                violated = np.where(current_pos < self.joint_limits_lower)[0]
                print(f"Warning: Current positions below lower limits for joints: {violated}")
        
        if self.joint_limits_upper is not None:
            if np.any(current_pos > self.joint_limits_upper):
                violated = np.where(current_pos > self.joint_limits_upper)[0]
                print(f"Warning: Current positions above upper limits for joints: {violated}")
    
    def _store_control_history(self, position_error: np.ndarray, velocity_error: np.ndarray,
                             desired_accel: np.ndarray, control_effort: float) -> None:
        """Store control data for research analysis."""
        if len(self.control_history) >= self.max_history_size:
            self.control_history.pop(0)  # Remove oldest entry
        
        self.control_history.append({
            'timestamp': time.time(),
            'position_error': position_error.copy(),
            'velocity_error': velocity_error.copy(),
            'desired_acceleration': desired_accel.copy(),
            'control_effort': control_effort,
            'kp': self.kp.copy(),
            'kd': self.kd.copy(),
            'rms_position_error': np.sqrt(np.mean(position_error**2)),
            'rms_velocity_error': np.sqrt(np.mean(velocity_error**2))
        })
    
    def update_gains(self, new_kp: Optional[Union[List[float], np.ndarray]] = None,
                    new_kd: Optional[Union[List[float], np.ndarray]] = None,
                    validate_bounds: bool = True) -> bool:
        """
        Update PD gains dynamically (for research applications).
        
        Args:
            new_kp: New proportional gains, optional
            new_kd: New derivative gains, optional
            validate_bounds: Whether to enforce safety bounds
            
        Returns:
            True if update successful, False otherwise
            
        Raises:
            ValueError: If new gains have wrong dimensions or are invalid
        """
        try:
            if new_kp is not None:
                new_kp_array = np.array(new_kp, dtype=np.float64)
                
                # Validate dimensions and values
                if new_kp_array.shape != (self.num_joints,):
                    raise ValueError(f"new_kp must have shape ({self.num_joints},)")
                if np.any(new_kp_array < 0):
                    raise ValueError("All Kp gains must be non-negative")
                
                # Apply safety bounds if requested
                if validate_bounds:
                    new_kp_array = np.clip(new_kp_array, self.kp_min, self.kp_max)
                
                # Store old value and update
                old_kp = self.kp.copy()
                self.kp = new_kp_array
                
                # Log adaptation
                self.adaptation_history.append({
                    'timestamp': time.time(),
                    'parameter': 'kp',
                    'old_value': old_kp,
                    'new_value': self.kp.copy(),
                    'bounded': validate_bounds
                })
            
            if new_kd is not None:
                new_kd_array = np.array(new_kd, dtype=np.float64)
                
                # Validate dimensions and values
                if new_kd_array.shape != (self.num_joints,):
                    raise ValueError(f"new_kd must have shape ({self.num_joints},)")
                if np.any(new_kd_array < 0):
                    raise ValueError("All Kd gains must be non-negative")
                
                # Apply safety bounds if requested
                if validate_bounds:
                    new_kd_array = np.clip(new_kd_array, self.kd_min, self.kd_max)
                
                # Store old value and update
                old_kd = self.kd.copy()
                self.kd = new_kd_array
                
                # Log adaptation
                self.adaptation_history.append({
                    'timestamp': time.time(),
                    'parameter': 'kd',
                    'old_value': old_kd,
                    'new_value': self.kd.copy(),
                    'bounded': validate_bounds
                })
            
            return True
            
        except Exception as e:
            print(f"Error updating gains: {e}")
            return False
    
    def get_gains(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current PD gains.
        
        Returns:
            Tuple of (kp, kd) as numpy arrays
        """
        return self.kp.copy(), self.kd.copy()
    
    def set_gain_bounds(self, kp_min: Optional[np.ndarray] = None, kp_max: Optional[np.ndarray] = None,
                       kd_min: Optional[np.ndarray] = None, kd_max: Optional[np.ndarray] = None) -> None:
        """
        Set bounds for gain adaptation (for research safety).
        
        Args:
            kp_min: Minimum allowed Kp gains
            kp_max: Maximum allowed Kp gains  
            kd_min: Minimum allowed Kd gains
            kd_max: Maximum allowed Kd gains
        """
        if kp_min is not None:
            self.kp_min = np.array(kp_min)
        if kp_max is not None:
            self.kp_max = np.array(kp_max)
        if kd_min is not None:
            self.kd_min = np.array(kd_min)
        if kd_max is not None:
            self.kd_max = np.array(kd_max)
    
    def get_control_statistics(self) -> Dict:
        """
        Get comprehensive control performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.control_history:
            return {"error": "No control history available"}
        
        # Extract data arrays
        position_errors = np.array([entry['position_error'] for entry in self.control_history])
        velocity_errors = np.array([entry['velocity_error'] for entry in self.control_history])
        control_efforts = np.array([entry['control_effort'] for entry in self.control_history])
        
        return {
            'num_samples': len(self.control_history),
            'total_control_calls': self.total_control_calls,
            'mean_position_error': np.mean(np.abs(position_errors), axis=0),
            'max_position_error': np.max(np.abs(position_errors), axis=0),
            'rms_position_error': np.sqrt(np.mean(position_errors**2, axis=0)),
            'mean_velocity_error': np.mean(np.abs(velocity_errors), axis=0),
            'max_velocity_error': np.max(np.abs(velocity_errors), axis=0),
            'rms_velocity_error': np.sqrt(np.mean(velocity_errors**2, axis=0)),
            'mean_control_effort': np.mean(control_efforts),
            'max_control_effort': np.max(control_efforts),
            'current_gains_kp': self.kp.copy(),
            'current_gains_kd': self.kd.copy(),
            'num_adaptations': len(self.adaptation_history)
        }
    
    def compute_control_effort(self, position_error: np.ndarray, velocity_error: np.ndarray) -> float:
        """
        Compute control effort magnitude for analysis.
        
        Args:
            position_error: Position tracking errors
            velocity_error: Velocity tracking errors
            
        Returns:
            Total control effort magnitude
        """
        return np.linalg.norm(self.kp * position_error) + np.linalg.norm(self.kd * velocity_error)
    
    def check_stability(self, mass_matrix_eigenvalues: Optional[np.ndarray] = None) -> Dict:
        """
        Perform basic stability analysis of current gains.
        
        Args:
            mass_matrix_eigenvalues: Eigenvalues of robot mass matrix, optional
            
        Returns:
            Dictionary with stability analysis results
        """
        stability_info = {
            'gains_positive': np.all(self.kp > 0) and np.all(self.kd > 0),
            'gains_reasonable': True,
            'warnings': []
        }
        
        # Check gain ratios
        gain_ratios = self.kd / np.sqrt(self.kp)
        if np.any(gain_ratios < 0.1):
            stability_info['warnings'].append("Some joints may be underdamped")
        if np.any(gain_ratios > 5.0):
            stability_info['warnings'].append("Some joints may be overdamped")
        
        # Advanced stability check if mass matrix provided
        if mass_matrix_eigenvalues is not None:
            min_mass_eigenval = np.min(mass_matrix_eigenvalues)
            damping_ratios = self.kd / (2 * np.sqrt(self.kp * min_mass_eigenval))
            
            stability_info['damping_ratios'] = damping_ratios
            stability_info['well_damped'] = np.all((damping_ratios > 0.3) & (damping_ratios < 1.5))
            
            if not stability_info['well_damped']:
                stability_info['warnings'].append("Damping ratios outside recommended range [0.3, 1.5]")
        
        return stability_info
    
    def reset_history(self) -> None:
        """Clear all tracking data (for new experiments)."""
        self.control_history.clear()
        self.adaptation_history.clear()
        self.total_control_calls = 0
        print("Control history reset")
    
    def save_history_to_file(self, filename: str) -> bool:
        """
        Save control history to file for analysis.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            data = {
                'control_history': self.control_history,
                'adaptation_history': self.adaptation_history,
                'statistics': self.get_control_statistics(),
                'controller_config': {
                    'num_joints': self.num_joints,
                    'initial_kp': self.kp.tolist(),
                    'initial_kd': self.kd.tolist(),
                    'torque_limits': self.torque_limits.tolist()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"History saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving history: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of controller."""
        return f"PDController(joints={self.num_joints}, active_time={time.time() - (self.last_control_time or time.time()):.1f}s)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"PDController(num_joints={self.num_joints}, "
                f"kp_range=[{np.min(self.kp):.2f}, {np.max(self.kp):.2f}], "
                f"kd_range=[{np.min(self.kd):.2f}, {np.max(self.kd):.2f}], "
                f"calls={self.total_control_calls})")