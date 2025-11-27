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

# Configuration import
import End_to_End_RL_In_Teleoperation.config.robot_config as cfg


class IKSolver:
    def __init__(self, model: mujoco.MjModel, joint_limits_lower, joint_limits_upper):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_limits_lower = joint_limits_lower.copy()
        self.joint_limits_upper = joint_limits_upper.copy()
        self.n_joints = len(joint_limits_lower)
        
        # Preallocate arrays
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.M_full = np.zeros((self.model.nv, self.model.nv))
        
        # Weights and Config
        self.virtual_inertia = np.array(cfg.IK_JOINT_WEIGHTS, dtype=np.float64)
        self.q_previous: Optional[NDArray[np.float64]] = cfg.INITIAL_JOINT_CONFIG.copy()
        
        # Define Rest Pose (using initial config or a custom comfortable pose)
        self.q_rest = cfg.INITIAL_JOINT_CONFIG.copy() 

    def _get_weighted_pseudoinverse_and_nullspace(self, J: NDArray, damping: float) -> Tuple[NDArray, NDArray]:
        """
        Calculates the Damped Least Squares Pseudoinverse and the Null-Space Projector.
        
        Formula:
        J_pinv = W^-1 J^T (J W^-1 J^T + lambda^2 I)^-1
        N      = I - J_pinv * J  
        """
        # Mass Matrix Retrieval
        mujoco.mj_fullM(self.model, self.M_full, self.data.qM)
        M = self.M_full[:self.n_joints, :self.n_joints]
       
        # Weight Matrix (Inertia + Virtual Weights)
        W = M + np.diag(self.virtual_inertia)
        try:
            W_inv = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            W_inv = np.eye(self.n_joints) # Fallback

        # Damped Least Squares Calculation
        lambda_sq = damping ** 2
        JWJt = J @ W_inv @ J.T
        A = JWJt + lambda_sq * np.eye(3)
        
        try:
            A_inv = np.linalg.solve(A, np.eye(3))
        except np.linalg.LinAlgError:
            # Handle numerical instability
            return np.zeros((self.n_joints, 3)), np.eye(self.n_joints)

        J_pinv = W_inv @ J.T @ A_inv
        
        # Null-Space Projector: N = I - J# J
        N = np.eye(self.n_joints) - J_pinv @ J
        
        return J_pinv, N

    def _solve_jacobian(self, target_pos, q_init, site_id):
        """
        Iterative IK with Adaptive Damping and Null-Space Bias.
        """
        
        q = q_init.copy()
        site_body_id = self.model.site_bodyid[site_id]
        
        # Adaptive Damping Initialization
        # We start with the configured damping, but reduce it if error is small
        current_damping = cfg.IK_JACOBIAN_DAMPING
        
        for i in range(cfg.IK_JACOBIAN_MAX_ITER):
            # 1. Forward Kinematics
            self.data.qpos[:self.n_joints] = q
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[site_id].copy()
            
            # 2. Error Calculation
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < cfg.IK_POSITION_TOLERANCE:
                return q, error_norm
            
            # 3. Adaptive Damping Logic
            # If error is very small (< 1mm), reduce damping to allow fine convergence
            if error_norm < 0.001: 
                adaptive_lambda = current_damping * 0.1
            else:
                adaptive_lambda = current_damping

            # 4. Jacobian Calculation
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, current_pos, site_body_id)
            J = self.jacp[:, :self.n_joints]
            
            # 5. Compute Inverse and Null-Space
            J_pinv, N = self._get_weighted_pseudoinverse_and_nullspace(J, adaptive_lambda)
            
            # 6. Primary Task (End-Effector Position)
            dq_main = J_pinv @ error
            
            # 7. Secondary Task (Null-Space Projection towards Rest Pose)
            # Objective: minimize 0.5 * || q - q_rest ||^2
            # Gradient: (q - q_rest)
            # We want to move OPPOSITE to the gradient: q_rest - q
            k_null = 0.5  # Gain for the null-space task
            dq_null = N @ (k_null * (self.q_rest - q))
            
            # 8. Update Joint State
            dq = dq_main + dq_null
            q = q + cfg.IK_JACOBIAN_STEP_SIZE * dq
            q = np.clip(q, self.joint_limits_lower, self.joint_limits_upper)
            
        return None, error_norm

    def _solve_optimization(self, target_pos, q_init, site_id):
        """
        If IK fails using the Jacobian method, fall back to optimization-based IK.
        Uses non-linear least squares to minimize position error and joint movement.
        
        formula: minimize ||pos_err|| + ||kinetic_proxy||
        where pos_err = target_pos - current_pos
              kinetic_proxy = (q - q_prev) * sqrt(M_diag) * continuity_gain
        """

        q_prev = self.q_previous if self.q_previous is not None else q_init
        self.data.qpos[:self.n_joints] = q_init
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_fullM(self.model, self.M_full, self.data.qM)
        M_diag = np.diag(self.M_full[:self.n_joints, :self.n_joints]) + self.virtual_inertia
        
        def cost_function(q):
            self.data.qpos[:self.n_joints] = q
            mujoco.mj_forward(self.model, self.data)
            curr_pos = self.data.site_xpos[site_id]
            pos_err = target_pos - curr_pos
            kinetic_proxy = (q - q_prev) * np.sqrt(M_diag) * cfg.IK_CONTINUITY_GAIN
            return np.concatenate([pos_err, kinetic_proxy])
        
        result = least_squares(cost_function, q_init, bounds=(self.joint_limits_lower, self.joint_limits_upper),
                               method='trf', max_nfev=cfg.IK_OPTIMIZATION_MAX_ITER, ftol=cfg.IK_OPT_FTOL, xtol=cfg.IK_OPT_XTOL)
        success = np.linalg.norm(result.fun[:3]) < cfg.IK_POSITION_TOLERANCE
        
        return (result.x, np.linalg.norm(result.fun[:3])) if success else (None, np.linalg.norm(result.fun[:3]))

    def solve(self, target_pos, q_init, body_name=None, tcp_offset=None, enforce_continuity=True, target_rot=None):
        try:
            site_id = self.model.site('panda_ee_site').id
        except KeyError:
            raise ValueError("Site 'panda_ee_site' not found")
        
        q_sol, error = self._solve_jacobian(target_pos, q_init, site_id)
        success = q_sol is not None

        if not success:
            q_sol, error = self._solve_optimization(target_pos, q_init, site_id)
            success = q_sol is not None
            if not success:
                print(f"IK FAILED COMPLETELY! Final position error: {error:.6f} m")
                return None, False, error

        # Continuity limiting
        if enforce_continuity and self.q_previous is not None:
            joint_change = q_sol - self.q_previous
            if np.linalg.norm(joint_change) > cfg.IK_MAX_JOINT_CHANGE:
                scale = cfg.IK_MAX_JOINT_CHANGE / np.linalg.norm(joint_change)
                old_q = q_sol.copy()
                q_sol = self.q_previous + scale * joint_change
                print(f"IK: Continuity limit applied → scaled joint change from {np.linalg.norm(joint_change):.4f} to {cfg.IK_MAX_JOINT_CHANGE:.4f}")

        self.q_previous = q_sol.copy()
        return q_sol, success, error

    def reset_trajectory(self, q_start: NDArray[np.float64] = None) -> None:
        """
        Reset the IK solver to a known initial configuration.
        """
        q_init = q_start.copy() if q_start is not None else cfg.INITIAL_JOINT_CONFIG.copy()
        self.q_previous = q_init.copy()
        self.data.qpos[:self.n_joints] = q_init
        self.data.qvel[:self.n_joints] = 0.0
        mujoco.mj_forward(self.model, self.data)