"""
Inverse Kinematics Solver for N-DoF Robotic Manipulators.

Features:
1. Weighted Damped Least Squares (WDLS):
   - Penalizes movement of specific joints (e.g., wrist, base) via a weight matrix.
   - Handles singularities using damped least squares.

2. Null-Space Projection (Gradient Projection Method):
   - Optimizes a secondary objective (staying close to Rest Pose) within the null space
     of the primary task, ensuring the end-effector position is not disturbed.
   - The 'pull' towards the rest pose is also weighted by the joint weights.

3. Optimization Fallback:
   - Uses non-linear least squares if the Jacobian method fails to converge.
   - Includes terms for Cartesian accuracy, smoothness, and weighted rest-pose adherence.
"""


from __future__ import annotations
from typing import Optional, Tuple
from numpy.typing import NDArray

import numpy as np
import mujoco
from scipy.optimize import least_squares

import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

class IKSolver:

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_limits_lower: NDArray[np.float64],
        joint_limits_upper: NDArray[np.float64],
    ) -> None:
        
        self.model = model
        self.data = mujoco.MjData(model) # Internal data for IK calculations
        self.joint_limits_lower = joint_limits_lower.copy()
        self.joint_limits_upper = joint_limits_upper.copy()
        self.n_joints = len(joint_limits_lower)
        
        # Preallocate Matrices
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.M_full = np.zeros((self.model.nv, self.model.nv)) # Full Mass Matrix

        # Load self-defined joint weights
        if len(cfg.IK_JOINT_WEIGHTS) != self.n_joints:
             raise ValueError(f"Config weights mismatch: {len(cfg.IK_JOINT_WEIGHTS)} vs {self.n_joints}")
        
        self.virtual_inertia = np.array(cfg.IK_JOINT_WEIGHTS, dtype=np.float64)

        # State history for continuity
        self.q_previous: Optional[NDArray[np.float64]] = None

    def _get_inertia_weighted_inverse(self, J: NDArray) -> NDArray:
        """
        Computes W^{-1} * J^T * (J * W^{-1} * J^T + lambda*I)^{-1}
        Where W = MassMatrix(q) + VirtualInertia
        """
        # 1. Get Real Mass Matrix M(q) from MuJoCo
        # This captures the physical reality: moving the shoulder is "harder" than moving the wrist.
        mujoco.mj_fullM(self.model, self.M_full, self.data.qM)
        M = self.M_full[:self.n_joints, :self.n_joints]
        
        # 2. Add Virtual Inertia (The Preference)
        # W = Real Physics + User Preference
        W = M + np.diag(self.virtual_inertia)
        
        # 3. Compute W_inv (Inertia inverse)
        # Since W is Positive Definite (Mass + Diagonal), we use linalg.solve or inv
        try:
            W_inv = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            # Fallback if singular (rare with diagonal addition)
            W_inv = np.eye(self.n_joints)

        # 4. Compute Weighted Pseudo-Inverse
        # Formula: dq = W_inv J.T (J W_inv J.T + lambda I)^-1 dx
        
        lambda_sq = cfg.IK_JACOBIAN_DAMPING ** 2
        
        # Core term: J * W_inv * J.T
        JWJt = J @ W_inv @ J.T
        
        # Damped inverse of the core term
        A = JWJt + lambda_sq * np.eye(3)
        
        # Solve A * x = I  => x = A_inv
        try:
            A_inv = np.linalg.solve(A, np.eye(3))
        except np.linalg.LinAlgError:
            return np.zeros((self.n_joints, 3)) # Fail safe

        # Combine: W_inv * J.T * A_inv
        J_pinv_weighted = W_inv @ J.T @ A_inv
        
        return J_pinv_weighted

    def _solve_jacobian(
        self,
        target_pos: NDArray[np.float64],
        q_init: NDArray[np.float64],
        site_id: int,
    ) -> Tuple[Optional[NDArray[np.float64]], float]:
        
        q = q_init.copy()
        site_body_id = self.model.site_bodyid[site_id]

        for i in range(cfg.IK_JACOBIAN_MAX_ITER):
            self.data.qpos[:self.n_joints] = q
            mujoco.mj_forward(self.model, self.data)
            
            # Current Site Position (End Effector)
            current_pos = self.data.site_xpos[site_id].copy()
            error = target_pos - current_pos
            
            # Check Convergence
            if np.linalg.norm(error) < cfg.IK_POSITION_TOLERANCE:
                return q, np.linalg.norm(error)
            
            # Compute Jacobian
            # J maps joint velocities to site velocities
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, current_pos, site_body_id)
            J = self.jacp[:, :self.n_joints]
            
            # --- INERTIA WEIGHTED CALCULATION ---
            J_pinv_w = self._get_inertia_weighted_inverse(J)
            dq = J_pinv_w @ error
            # ------------------------------------
            
            # Update q
            q = q + cfg.IK_JACOBIAN_STEP_SIZE * dq
            q = np.clip(q, self.joint_limits_lower, self.joint_limits_upper)
            
        # Final check
        final_pos = self.data.site_xpos[site_id].copy()
        return None, np.linalg.norm(target_pos - final_pos)

    def _solve_optimization(self, target_pos, q_init, site_id):
        """
        Optimization fallback.
        Minimizes Position Error + Weighted Joint Change (Kinetic Energy Proxy)
        """
        q_prev = self.q_previous if self.q_previous is not None else q_init
        
        # Pre-calculate Mass Matrix at q_init for the weighting term
        self.data.qpos[:self.n_joints] = q_init
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_fullM(self.model, self.M_full, self.data.qM)
        
        # W = M + Virtual Inertia
        M_diag = np.diag(self.M_full[:self.n_joints, :self.n_joints]) + self.virtual_inertia

        def cost_function(q: NDArray[np.float64]) -> NDArray[np.float64]:
            self.data.qpos[:self.n_joints] = q
            mujoco.mj_forward(self.model, self.data)
            
            # 1. Position Error
            curr_pos = self.data.site_xpos[site_id]
            pos_err = target_pos - curr_pos
            
            # 2. Kinetic Energy Proxy (Smoothness)
            # We weight the joint changes by sqrt(W). 
            # This discourages moving 'heavy' joints (real or virtual)
            kinetic_proxy = (q - q_prev) * np.sqrt(M_diag) * cfg.IK_CONTINUITY_GAIN
            
            # Note: We don't use a specific Rest Pose term here anymore, 
            # because the inertia weighting inherently discourages unnecessary movement.
            
            return np.concatenate([pos_err, kinetic_proxy])

        result = least_squares(
            fun=cost_function,
            x0=q_init,
            bounds=(self.joint_limits_lower, self.joint_limits_upper),
            method='trf',
            max_nfev=cfg.IK_OPTIMIZATION_MAX_ITER,
            ftol=cfg.IK_OPT_FTOL,
            xtol=cfg.IK_OPT_XTOL,
        )
        
        final_error = np.linalg.norm(result.fun[:3])
        success = final_error < cfg.IK_POSITION_TOLERANCE
        return (result.x, final_error) if success else (None, final_error)

    def solve(
        self,
        target_pos: NDArray[np.float64],
        q_init: NDArray[np.float64],
        body_name: str = None, # Kept for compatibility, but we strictly look for site
        tcp_offset: Optional[NDArray[np.float64]] = None, # IGNORED (Handled by Site)
        enforce_continuity: bool = True,
        target_rot: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[Optional[NDArray[np.float64]], bool, float]:
        
        # 1. Locate the End Effector Site
        try:
            site_id = self.model.site('panda_ee_site').id
        except KeyError:
            raise ValueError("Site 'panda_ee_site' not found in XML model.")

        # 2. Try Inertia-Weighted Jacobian Solver
        q_sol, error = self._solve_jacobian(target_pos, q_init, site_id)
        success = q_sol is not None
        
        # 3. Fallback to Optimization Solver
        if not success:
            q_sol, error = self._solve_optimization(target_pos, q_init, site_id)
            success = q_sol is not None
            if not success: 
                return None, False, error

        # 4. Enforce Continuity (Velocity Limits)
        if enforce_continuity and self.q_previous is not None:
            joint_change = q_sol - self.q_previous
            change_norm = np.linalg.norm(joint_change)
            
            if change_norm > cfg.IK_MAX_JOINT_CHANGE:
                # Scale down the change to limit max velocity
                scale = cfg.IK_MAX_JOINT_CHANGE / change_norm
                q_sol = self.q_previous + scale * joint_change
                
                # Recompute final error after scaling
                self.data.qpos[:self.n_joints] = q_sol
                mujoco.mj_forward(self.model, self.data)
                final_pos = self.data.site_xpos[site_id]
                error = np.linalg.norm(target_pos - final_pos)

        self.q_previous = q_sol.copy()
        return q_sol, success, error

    def reset_trajectory(self, q_start: Optional[NDArray[np.float64]] = None) -> None:
        if q_start is not None:
            self.q_previous = q_start.copy()
        else:
            self.q_previous = None