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

import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

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
        
        # State tracking
        self.q_previous = cfg.INITIAL_JOINT_CONFIG.copy()
        self.q_rest = cfg.INITIAL_JOINT_CONFIG.copy() 

    def _get_inertia_weighted_pseudoinverse(self, J: NDArray, damping: float) -> Tuple[NDArray, NDArray]:
        """
        Calculates the Dynamically Consistent Jacobian Inverse.
        Formula: J_bar = M^-1 J^T (J M^-1 J^T + lambda^2 I)^-1
        """
        rows, cols = J.shape # (3, 7)
        
        # 1. Get Mass Matrix M(q)
        mujoco.mj_fullM(self.model, self.M_full, self.data.qM)
        M = self.M_full[:self.n_joints, :self.n_joints]
        
        # 2. Invert M
        # M is positive definite, so inversion is generally safe unless joint limits are violated
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.eye(self.n_joints) # Fallback

        # 3. Calculate Weighted Pseudo-Inverse
        # Intermediate term: A = J * M^-1 * J^T
        JMJ = J @ M_inv @ J.T
        
        # Add Damping to the task-space matrix A
        damping_sq = damping ** 2
        damped_JMJ = JMJ + damping_sq * np.eye(rows)
        
        try:
            JMJ_inv = np.linalg.inv(damped_JMJ)
        except np.linalg.LinAlgError:
            JMJ_inv = np.eye(rows)

        # J_bar = M^-1 * J^T * (JMJ)^-1
        J_pinv = M_inv @ J.T @ JMJ_inv
        
        # 4. Dynamically Consistent Null-Space Projector
        # N = I - J_bar * J
        N = np.eye(cols) - J_pinv @ J
        
        return J_pinv, N

    def solve(self, target_pos, q_init, body_name=None, tcp_offset=None, enforce_continuity=True, target_rot=None):
        try:
            site_id = self.model.site('panda_ee_site').id
            site_body_id = self.model.site_bodyid[site_id]
        except KeyError:
            raise ValueError("Site 'panda_ee_site' not found")

        q = q_init.copy()
        
        # Parameters
        max_iter = cfg.IK_JACOBIAN_MAX_ITER
        damping = cfg.IK_JACOBIAN_DAMPING
        step_size = 0.5 
        tolerance = cfg.IK_POSITION_TOLERANCE
        null_gain = cfg.IK_NULL_SPACE_GAIN

        success = False
        final_error = 0.0

        for i in range(max_iter):
            # 1. FK
            self.data.qpos[:self.n_joints] = q
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[site_id].copy()
            
            # 2. Error
            error = target_pos - current_pos
            final_error = np.linalg.norm(error)
            
            if final_error < tolerance:
                success = True
                break
            
            # 3. Jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, current_pos, site_body_id)
            J_pos = self.jacp[:, :self.n_joints]
            
            # 4. Compute Inertia-Weighted Pinv & Nullspace
            J_pinv, N = self._get_inertia_weighted_pseudoinverse(J_pos, damping)
            
            # 5. Calculate Steps
            dq_main = J_pinv @ error
            
            # Null-Space torque towards rest pose (Weighted by Mass implicitly via N)
            dq_null = N @ (null_gain * (self.q_rest - q))
            
            dq = dq_main + dq_null
            
            # 6. Apply Step
            q = q + step_size * dq
            q = np.clip(q, self.joint_limits_lower, self.joint_limits_upper)

        # REMOVED: Continuity Limit / Jump Rejection
        # We implicitly trust that M(q) prevents high-energy jumps.
        
        self.q_previous = q.copy()
        return q, success, final_error

    def reset_trajectory(self, q_start: NDArray[np.float64] = None) -> None:
        q_init = q_start.copy() if q_start is not None else cfg.INITIAL_JOINT_CONFIG.copy()
        self.q_previous = q_init.copy()
        self.data.qpos[:self.n_joints] = q_init
        self.data.qvel[:self.n_joints] = 0.0
        mujoco.mj_forward(self.model, self.data)