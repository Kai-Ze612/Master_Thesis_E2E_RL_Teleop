"""
Calculate the inverse kinematics for N-dof robotic manipulators.

First, apply Jacobian based IK solver.
if fails, apply optimization based IK solver.
"""

from __future__ import annotations
from numpy.typing import NDArray
from dataclasses import dataclass
from abc import ABC, abstractmethod

import mujoco
from scipy.optimize import minimize

import numpy as np

@dataclass(frozen=True)
class IKsolver:
    """Inverse kinematics solver parameters."""
    def __init__(
        self,
        model: mujoco.MjModel,
        joint_limits_lower: NDArray[np.float64],
        joint_limits_upper: NDArray[np.float64],
        jacobian_max_iter: int = 50,
        jacobian_tolerance: float = 1e-4,
        jacobian_step_size: float = 0.5,
        jacobian_damping: float = 1e-4,
        optimization_max_iter: int = 100,
        optimization_tolerance: float = 1e-6,
    ) -> None:

        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_limits_lower = joint_limits_lower.copy()
        self.joint_limits_upper = joint_limits_upper.copy()
        self.n_joints = len(joint_limits_lower)
        
        # Jacobian parameters
        self.jacobian_max_iter = jacobian_max_iter
        self.jacobian_tolerance = jacobian_tolerance
        self.jacobian_step_size = jacobian_step_size
        self.jacobian_damping = jacobian_damping
        
        # Optimization parameters
        self.optimization_max_iter = optimization_max_iter
        self.optimization_tolerance = optimization_tolerance
        
        # Preallocate Jacobian matrices
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        
        # Statistics
        self.last_method_used = "none"
        self.jacobian_success_count = 0
        self.optimization_success_count = 0
        self.total_solve_count = 0

    def _solve_jacobian(
        self,
        target_pos: NDArray[np.float64],
        q_init: NDArray[np.float64],
        body_id: int,
        tcp_offset: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[Optional[NDArray[np.float64]], bool]:
        """Internal Jacobian solver."""
        q = q_init.copy()
        
        for iteration in range(self.jacobian_max_iter):
            self.data.qpos[:self.n_joints] = q
            mujoco.mj_forward(self.model, self.data)
            
            # Compute current position (with or without TCP offset)
            if tcp_offset is None:
                current_pos = self.data.xpos[body_id].copy()
                point_for_jacobian = current_pos
            else:
                body_pos = self.data.xpos[body_id].copy()
                body_rot = self.data.xmat[body_id].reshape(3, 3)
                current_pos = body_pos + body_rot @ tcp_offset
                point_for_jacobian = current_pos
            
            # Position error
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < self.jacobian_tolerance:
                return q, True
            
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
            
            # Damped pseudo-inverse
            JJT = J @ J.T
            lambda_squared = self.jacobian_damping ** 2
            J_pinv = J.T @ np.linalg.inv(JJT + lambda_squared * np.eye(3))
            
            # Update
            dq = self.jacobian_step_size * J_pinv @ error
            q = q + dq
            q = np.clip(q, self.joint_limits_lower, self.joint_limits_upper)
        
        return None, False

    def _solve_optimization(
        self,
        target_pos: NDArray[np.float64],
        q_init: NDArray[np.float64],
        body_id: int,
        tcp_offset: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[Optional[NDArray[np.float64]], bool]:
        """Internal optimization solver."""
        if tcp_offset is None:
            def cost_function(q: NDArray[np.float64]) -> NDArray[np.float64]:
                self.data.qpos[:self.n_joints] = q
                mujoco.mj_forward(self.model, self.data)
                current_pos = self.data.xpos[body_id].copy()
                return target_pos - current_pos
        else:
            def cost_function(q: NDArray[np.float64]) -> NDArray[np.float64]:
                self.data.qpos[:self.n_joints] = q
                mujoco.mj_forward(self.model, self.data)
                body_pos = self.data.xpos[body_id].copy()
                body_rot = self.data.xmat[body_id].reshape(3, 3)
                tcp_pos = body_pos + body_rot @ tcp_offset
                return target_pos - tcp_pos
        
        result = least_squares(
            fun=cost_function,
            x0=q_init,
            bounds=(self.joint_limits_lower, self.joint_limits_upper),
            method='trf',
            max_nfev=self.optimization_max_iter,
        )
        
        final_error = np.linalg.norm(result.fun)
        success = final_error < self.optimization_tolerance
        
        return (result.x, True) if success else (None, False)

    def solve(
        self,
        target_pos: NDArray[np.float64],
        q_init: NDArray[np.float64],
        body_name: str,
    ) -> Tuple[Optional[NDArray[np.float64]], bool]:
        """Solve IK using hybrid method.
        
        Tries Jacobian first, falls back to optimization if needed.
        
        Args:
            target_pos: Target position (m) - shape (3,)
            q_init: Initial joint config (rad) - shape (n,)
            body_name: End-effector body name
            
        Returns:
            (q_solution, success) - solution and success flag
        """
        self.total_solve_count += 1
        
        try:
            body_id = self.model.body(body_name).id
        except KeyError:
            self.last_method_used = "error"
            return None, False
        
        # Try Jacobian first (fast!)
        q_jacobian, jacobian_success = self._solve_jacobian(
            target_pos, q_init, body_id, tcp_offset=None
        )
        
        if jacobian_success:
            self.last_method_used = "jacobian"
            self.jacobian_success_count += 1
            return q_jacobian, True
        
        # Fallback to optimization (robust!)
        q_opt, opt_success = self._solve_optimization(
            target_pos, q_init, body_id, tcp_offset=None
        )
        
        if opt_success:
            self.last_method_used = "optimization"
            self.optimization_success_count += 1
            return q_opt, True
        
        self.last_method_used = "failed"
        return None, False

    def solve_with_tcp_offset(
        self,
        target_pos: NDArray[np.float64],
        q_init: NDArray[np.float64],
        body_name: str,
        tcp_offset: NDArray[np.float64],
    ) -> Tuple[Optional[NDArray[np.float64]], bool]:
        """Solve IK with TCP offset using hybrid method.
        
        Args:
            target_pos: Target TCP position (m) - shape (3,)
            q_init: Initial joint config (rad) - shape (n,)
            body_name: End-effector body name
            tcp_offset: Offset to TCP in body frame (m) - shape (3,)
            
        Returns:
            (q_solution, success) - solution and success flag
        """
        self.total_solve_count += 1
        
        try:
            body_id = self.model.body(body_name).id
        except KeyError:
            self.last_method_used = "error"
            return None, False
        
        # Try Jacobian first
        q_jacobian, jacobian_success = self._solve_jacobian(
            target_pos, q_init, body_id, tcp_offset=tcp_offset
        )
        
        if jacobian_success:
            self.last_method_used = "jacobian"
            self.jacobian_success_count += 1
            return q_jacobian, True
        
        # Fallback to optimization
        q_opt, opt_success = self._solve_optimization(
            target_pos, q_init, body_id, tcp_offset=tcp_offset
        )
        
        if opt_success:
            self.last_method_used = "optimization"
            self.optimization_success_count += 1
            return q_opt, True
        
        self.last_method_used = "failed"
        return None, False

    def get_statistics(self) -> dict:
        """Get solver statistics.
        
        Returns:
            Dictionary with solver statistics
        """
        if self.total_solve_count == 0:
            return {
                "total_solves": 0,
                "jacobian_success_rate": 0.0,
                "optimization_success_rate": 0.0,
                "overall_success_rate": 0.0,
            }
        
        return {
            "total_solves": self.total_solve_count,
            "jacobian_successes": self.jacobian_success_count,
            "optimization_successes": self.optimization_success_count,
            "jacobian_success_rate": self.jacobian_success_count / self.total_solve_count,
            "optimization_success_rate": self.optimization_success_count / self.total_solve_count,
            "overall_success_rate": (self.jacobian_success_count + self.optimization_success_count) / self.total_solve_count,
        }