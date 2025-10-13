"""
Calculate the inverse kinematics for N-dof robotic manipulators.

Uses a hybrid approach:
- First, apply Jacobian based IK solver:
    - q_dot = J_pseudo_inverse * position_error
    - Use damped least squares to compute pseudo-inverse.
    - Use null-space projection to enforce trajectory continuity.
- If it fails, apply optimization based IK solver. (non-linear least squares)
"""

from __future__ import annotations
from typing import Optional, Tuple
from numpy.typing import NDArray

import numpy as np
import mujoco
from scipy.optimize import least_squares


class IKSolver:

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_limits_lower: NDArray[np.float64],
        joint_limits_upper: NDArray[np.float64],
        jacobian_max_iter: int = 100,
        position_tolerance: float = 1e-4,
        jacobian_step_size: float = 0.25,
        jacobian_damping: float = 1e-4,
        optimization_max_iter: int = 100,
        optimization_ftol: float = 1e-8,
        optimization_xtol: float = 1e-8,
        max_joint_change: float = 0.1,
        continuity_gain: float = 0.5,
    ) -> None:
        
        # Model configuration
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_limits_lower = joint_limits_lower.copy()
        self.joint_limits_upper = joint_limits_upper.copy()
        self.n_joints = len(joint_limits_lower)
        
        # Trajectory continuity parameters
        self.max_joint_change = max_joint_change
        self.continuity_gain = np.clip(continuity_gain, 0.0, 1.0)
        self.q_previous = None
        
        # Jacobian solver parameters
        self.jacobian_max_iter = jacobian_max_iter
        self.position_tolerance = position_tolerance
        self.jacobian_step_size = jacobian_step_size
        self.jacobian_damping = jacobian_damping
        
        # Optimization solver parameters
        self.optimization_max_iter = optimization_max_iter
        self.optimization_ftol = optimization_ftol
        self.optimization_xtol = optimization_xtol
        
        # Preallocate Jacobian matrices
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

    def _compute_current_position(
        self,
        q: NDArray[np.float64],
        body_id: int,
        tcp_offset: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """ Using forward kinematics to compute current TCP position."""
        self.data.qpos[:self.n_joints] = q
        mujoco.mj_forward(self.model, self.data)
        
        body_pos = self.data.xpos[body_id].copy()
        
        if tcp_offset is None:
            return body_pos
        else:
            body_rot = self.data.xmat[body_id].reshape(3, 3)
            return body_pos + body_rot @ tcp_offset

    def _solve_jacobian(
        self,
        target_pos: NDArray[np.float64],
        q_init: NDArray[np.float64],
        body_id: int,
        tcp_offset: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[Optional[NDArray[np.float64]], float]:
        
        q = q_init.copy()
        q_reference = self.q_previous if self.q_previous is not None else q_init
        
        for iteration in range(self.jacobian_max_iter):
            self.data.qpos[:self.n_joints] = q
            mujoco.mj_forward(self.model, self.data)
            
            # Get body pose
            body_pos = self.data.xpos[body_id].copy()
            body_rot = self.data.xmat[body_id].reshape(3, 3)
            
            # Compute current TCP position
            if tcp_offset is None:
                current_pos = body_pos
                point_for_jacobian = body_pos
            else:
                current_pos = body_pos + body_rot @ tcp_offset
                point_for_jacobian = current_pos
            
            # Position error
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            # Check convergence
            if error_norm < self.position_tolerance:
                return q, error_norm
            
            # Compute Jacobian
            mujoco.mj_jac(
                self.model,
                self.data,
                self.jacp,
                self.jacr,
                point_for_jacobian,
                body_id
            )
            
            J = self.jacp[:, :self.n_joints]
            
            # Damped pseudo-inverse (numerically stable)
            lambda_squared = self.jacobian_damping ** 2
            JJT_damped = J @ J.T + lambda_squared * np.eye(3)
            
            try:
                J_pinv = J.T @ np.linalg.solve(JJT_damped, np.eye(3))
            except np.linalg.LinAlgError:
                return None, error_norm
            
            # Primary task: position error
            dq_primary = J_pinv @ error
            
            # Secondary task: stay close to reference
            null_space_projector = np.eye(self.n_joints) - J_pinv @ J
            dq_continuity = q_reference - q
            dq_secondary = null_space_projector @ dq_continuity
            
            # Combined update with continuity consideration
            dq = dq_primary + self.continuity_gain * dq_secondary
            
            # Update with step size and joint limits
            q = q + self.jacobian_step_size * dq
            q = np.clip(q, self.joint_limits_lower, self.joint_limits_upper)
        
        # Did not converge - compute final error
        final_pos = self._compute_current_position(q, body_id, tcp_offset)
        final_error = np.linalg.norm(target_pos - final_pos)
        return None, final_error

    def _solve_optimization(
        self,
        target_pos: NDArray[np.float64],
        q_init: NDArray[np.float64],
        body_id: int,
        tcp_offset: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[Optional[NDArray[np.float64]], float]:
        
        q_reference = self.q_previous if self.q_previous is not None else q_init
        
        def cost_function(q: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compute position residual."""
            
            current_pos = self._compute_current_position(q, body_id, tcp_offset)
            position_error = target_pos - current_pos
            
            # Add continuity term: penalize deviation from previous config
            continuity_error = self.continuity_gain * 0.05 * (q - q_reference)
            
            return np.concatenate([position_error, continuity_error])
            
        result = least_squares(
            fun=cost_function,
            x0=q_init,
            bounds=(self.joint_limits_lower, self.joint_limits_upper),
            method='trf',
            max_nfev=self.optimization_max_iter,
            ftol=self.optimization_ftol,
            xtol=self.optimization_xtol,
        )
        
        # Check only position error (first 3 elements)
        final_error = np.linalg.norm(result.fun[:3])
        success = final_error < self.position_tolerance

        return (result.x, final_error) if success else (None, final_error)

    def solve(
        self,
        target_pos: NDArray[np.float64],
        q_init: NDArray[np.float64],
        body_name: str,
        tcp_offset: Optional[NDArray[np.float64]] = None,
        enforce_continuity: bool = True,
    ) -> Tuple[Optional[NDArray[np.float64]], bool, float]:
        
        # Validate inputs
        if target_pos.shape != (3,):
            raise ValueError(f"target_pos must be shape (3,), got {target_pos.shape}")
        if q_init.shape != (self.n_joints,):
            raise ValueError(
                f"q_init must be shape ({self.n_joints},), got {q_init.shape}"
            )
        if tcp_offset is not None and tcp_offset.shape != (3,):
            raise ValueError(f"tcp_offset must be shape (3,), got {tcp_offset.shape}")
        
        # Get body ID
        try:
            body_id = self.model.body(body_name).id
        except KeyError:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
        
        # Try Jacobian first
        q_jacobian, jacobian_error = self._solve_jacobian(
            target_pos, q_init, body_id, tcp_offset
        )
        
        if q_jacobian is not None:
            q_solution = q_jacobian
            error = jacobian_error
            success = True
        else:
            # Fallback to optimization
            q_opt, opt_error = self._solve_optimization(
                target_pos, q_init, body_id, tcp_offset
            )
            
            if q_opt is not None:
                q_solution = q_opt
                error = opt_error
                success = True
            else:
                return None, False, opt_error
        
        # Enforce maximum joint change constraint
        if enforce_continuity and self.q_previous is not None:
            joint_change = q_solution - self.q_previous
            change_magnitude = np.linalg.norm(joint_change)
            
            if change_magnitude > self.max_joint_change:
                # Scale down the change
                scale = self.max_joint_change / change_magnitude
                q_solution = self.q_previous + scale * joint_change
                
                # Recompute error after scaling
                final_pos = self._compute_current_position(q_solution, body_id, tcp_offset)
                error = np.linalg.norm(target_pos - final_pos)
        
        # Update previous solution
        self.q_previous = q_solution.copy()
        
        return q_solution, success, error

    def reset_trajectory(self, q_start: Optional[NDArray[np.float64]] = None) -> None:
        if q_start is not None:
            if q_start.shape != (self.n_joints,):
                raise ValueError(
                    f"q_start must be shape ({self.n_joints},), got {q_start.shape}"
                )
            self.q_previous = q_start.copy()
        else:
            self.q_previous = None