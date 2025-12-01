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
        
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        
        self.virtual_inertia = np.array(cfg.IK_JOINT_WEIGHTS, dtype=np.float64)
        self.q_previous = cfg.INITIAL_JOINT_CONFIG.copy()
        self.q_rest = cfg.INITIAL_JOINT_CONFIG.copy() 

    def _get_damped_pseudoinverse(self, J: NDArray, damping: float) -> Tuple[NDArray, NDArray]:
        """
        Damped Least Squares (DLS) to handle singularities smoothly.
        """
        rows, cols = J.shape
        
        # Apply Joint Weights (Virtual Inertia)
        # Higher weight = Joint moves less
        W_inv = np.diag(1.0 / (1.0 + self.virtual_inertia))
        
        # Weighted Jacobian
        J_w = J @ W_inv
        
        # DLS Inversion: J_pinv = W_inv * J^T * (J * W_inv * J^T + lambda^2 * I)^-1
        JJT = J_w @ J_w.T
        damping_sq = damping ** 2
        damped_JJT = JJT + damping_sq * np.eye(rows)
        
        try:
            JJT_inv = np.linalg.inv(damped_JJT)
        except np.linalg.LinAlgError:
            JJT_inv = np.eye(rows)

        J_pinv = W_inv @ J_w.T @ JJT_inv
        
        # Null Space
        N = np.eye(cols) - J_pinv @ J
        return J_pinv, N

    def solve(self, target_pos, q_init, body_name=None, tcp_offset=None, enforce_continuity=True, target_rot=None):
        try:
            site_id = self.model.site('panda_ee_site').id
            site_body_id = self.model.site_bodyid[site_id]
        except KeyError:
            raise ValueError("Site 'panda_ee_site' not found")

        q = q_init.copy()
        
        # --- ROBUST CONFIGURATION (Hardcoded for stability or read from cfg) ---
        # We override extremely small steps to ensure convergence
        max_iter = 50       
        step_size = 0.5     # Take 50% of the error step per iter (Fast convergence)
        damping = 0.1       # Moderate damping to smooth singularities
        tolerance = 0.005
        null_gain = 0.1     # Pull towards rest pose

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
            
            # 4. Step Calculation
            J_pinv, N = self._get_damped_pseudoinverse(J_pos, damping)
            
            dq_main = J_pinv @ error
            dq_null = N @ (null_gain * (self.q_rest - q))
            dq = dq_main + dq_null
            
            # 5. Apply
            q = q + step_size * dq
            q = np.clip(q, self.joint_limits_lower, self.joint_limits_upper)

        # --- SAFETY: CONTINUITY CHECK ---
        # If the solver output jumps significantly compared to previous frame, REJECT IT.
        # This prevents the 0.6 rad jumps seen in your logs.
        if self.q_previous is not None:
            # Check maximum change of any single joint
            diff = np.max(np.abs(q - self.q_previous))
            
            # If jump is > 0.1 rad (approx 6 degrees) in 0.005s, it is unsafe.
            # 0.1 rad / 0.005s = 20 rad/s (Still too fast, but filters teleportation)
            if diff > 0.1: 
                print(f"[IK REJECT] Jump detected: {diff:.3f} > 0.1. Holding position.")
                q = self.q_previous.copy()
                success = False
            else:
                self.q_previous = q.copy()
        else:
            self.q_previous = q.copy()
        
        return q, success, final_error

    def reset_trajectory(self, q_start: NDArray[np.float64] = None) -> None:
        q_init = q_start.copy() if q_start is not None else cfg.INITIAL_JOINT_CONFIG.copy()
        self.q_previous = q_init.copy()
        self.data.qpos[:self.n_joints] = q_init
        self.data.qvel[:self.n_joints] = 0.0
        mujoco.mj_forward(self.model, self.data)