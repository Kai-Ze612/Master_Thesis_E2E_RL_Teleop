"""
Calculate for Inverse Kinematics using optimization algorithms
1. Jacobian IK
2. Optimization IK
"""

"""
This is a generalized IK solver, it reads the len of joint_limits to determine the number of joints
"""

import numpy as np
import mujoco
from scipy.optimize import minimize


class InverseKinematicsSolver:
    def __init__(self, model, joint_limits_lower, joint_limits_upper):
        # Initial parameters here
       
        # Loading Mujoco model
        self.model = model
        self.data = mujoco.MjData(self.model) 
        
        # Joint limits
        self.joint_limits_lower = np.array(joint_limits_lower)
        self.joint_limits_upper = np.array(joint_limits_upper)
        self.bounds = list(zip(self.joint_limits_lower, self.joint_limits_upper))
        
        # Determine Number of Joints
        self.num_joints = len(self.joint_limits_lower)
        
        ## OPTIMIZATION: Joint prioritization - penalize wrist movement more heavily
        # Lower weights mean less movement preference for those joints
        # This is customized for the project with Franka Panda 
        if self.num_joints == 7:  # Franka Panda arm
            self.joint_weights = np.array([1.0, 1.0, 0.8, 0.5, 0.5, 0.3, 0.8])  # Heavily penalize joints 6,7
        else:
            self.joint_weights = np.ones(self.num_joints)
       
        # Jacobian parameters
        self.jacobian_max_iterations = 30
        self.jacobian_tolerance = 1e-3
        self.jacobian_step_size = 0.5
        self.jacobian_damping = 0.01
        
        # Damping for joint movements to prevent instability
        if self.num_joints == 7:
            self.joint_damping = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1])  # Higher damping for wrist
        else:
            self.joint_damping = np.full(self.num_joints, 0.01)
        
        # Optimization parameters
        self.optimization_tolerance = 0.02
        self.optimization_max_iterations = 100

    def solver(self, target_position, current_joint_states, ee_body_name):
        
        # Convert inputs to numpy arrays
        current_joint_states = np.array(current_joint_states)
        target_position = np.array(target_position[:3])
        
        # Try Jacobian method first
        jacobian_solution, error = self._solve_jacobian(target_position, current_joint_states, ee_body_name)

        if jacobian_solution is not None:
            return jacobian_solution, error
        
        # Fallback to optimization method
        optimization_solution, error = self._solve_optimization(target_position, current_joint_states, ee_body_name)

        return optimization_solution, error

    def _solve_jacobian(self, target_position, initial_guess, ee_body_name):
        """Solve using weighted Jacobian method with joint prioritization."""
        q = np.array(initial_guess)
        target_pos = np.array(target_position[:3])
        ee_id = self.model.body(ee_body_name).id        

        for iteration in range(self.jacobian_max_iterations):
            # Forward kinematics
            self.data.qpos[:self.num_joints] = q
            mujoco.mj_forward(self.model, self.data)
            
            # Position error
            current_pos = self.data.xpos[ee_id]
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)
            
            # Check convergence
            if error_norm < self.jacobian_tolerance:
                return q, error_norm

            # Compute Jacobian
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, None, ee_id)
            jacobian = jacp[:, :self.num_joints]

            # Scale columns of Jacobian by joint weights (lower weight = less movement)
            weighted_jacobian = jacobian * self.joint_weights[np.newaxis, :]
            
            # Weighted damped least squares solution
            try:
                # Create diagonal damping matrix based on joint priorities
                damping_matrix = np.diag(self.joint_damping)
                
                # Compute weighted pseudo-inverse
                JJT_damped = weighted_jacobian @ weighted_jacobian.T + self.jacobian_damping * np.eye(3)
                jacobian_pinv = weighted_jacobian.T @ np.linalg.inv(JJT_damped)
                
                # Compute joint updates with additional regularization
                dq_raw = self.jacobian_step_size * jacobian_pinv @ pos_error
                
                # Additional penalty to reduce wrist joint movements
                # Further reduce wrist joint movements
                if self.num_joints == 7:
                    dq_raw[5] *= 0.3  # Joint 6 (panda_joint6)
                    dq_raw[6] *= 0.3  # Joint 7 (panda_joint7)
                
                # Apply updates with joint limits
                q_new = q + dq_raw
                q_new = np.clip(q_new, self.joint_limits_lower, self.joint_limits_upper)
                
                q = q_new
                
            except np.linalg.LinAlgError:
                # Singular Jacobian - method failed
                return None, None
        
        # Max iterations reached - check if solution is acceptable
        self.data.qpos[:self.num_joints] = q
        mujoco.mj_forward(self.model, self.data)
        current_pos = self.data.xpos[ee_id]
        final_error = np.linalg.norm(target_pos - current_pos)
        
        if final_error < self.optimization_tolerance:  # Use optimization tolerance as fallback
            return q, final_error
        
        return None, None

    def _solve_optimization(self, target_position, initial_guess, ee_body_name):
        """Solve using optimization method with joint movement penalties."""
        ee_id = self.model.body(ee_body_name).id
        initial_guess = np.array(initial_guess)

        def objective_function(joint_angles):
            self.data.qpos[:self.num_joints] = joint_angles
            mujoco.mj_forward(self.model, self.data)
            current_ee_pos = self.data.xpos[ee_id]
            
            # Primary objective: position error
            position_error = np.linalg.norm(current_ee_pos - target_position)
            
            # Secondary objective: minimize joint movement from initial guess
            joint_movement = joint_angles - initial_guess
            
            # Apply joint-specific penalties (higher penalty for wrist joints)
            movement_penalty = np.sum((joint_movement ** 2) / self.joint_weights)
            
            # Combined objective (position error dominates, but joint movement matters)
            total_cost = position_error + 0.01 * movement_penalty
            
            return total_cost

        # Constraint to keep wrist joints close to initial values
        def wrist_constraint(joint_angles):
            if self.num_joints == 7:
                # Constraint: wrist joints should not move more than 0.3 radians
                wrist_movement = np.abs(joint_angles[5:7] - initial_guess[5:7])
                max_wrist_movement = 0.3
                return max_wrist_movement - np.max(wrist_movement)
            return 1.0  # Always satisfied for non-7DOF arms
        
        constraints = {'type': 'ineq', 'fun': wrist_constraint}
        
        try:
            result = minimize(
                objective_function,
                initial_guess,
                method='SLSQP',
                bounds=self.bounds,
                constraints=constraints,
                options={'maxiter': self.optimization_max_iterations, 'ftol': 1e-6}
            )
            
            if result.success:
                self.data.qpos[:self.num_joints] = result.x
                mujoco.mj_forward(self.model, self.data)
                current_ee_pos = self.data.xpos[ee_id]
                position_error = np.linalg.norm(current_ee_pos - target_position)
                
                if position_error < self.optimization_tolerance:
                    return result.x, position_error
            
            return None, None

        except Exception:
            return None, None

    # Method to limit wrist movement post-optimization
    def _limit_wrist_movement(self, solution, initial_guess, max_wrist_change=0.2):
        """Limit wrist joint movement to prevent instability."""
        if solution is None or self.num_joints != 7:
            return solution
            
        limited_solution = solution.copy()
        
        # Limit joints 5 and 6 (indices 5, 6) movement
        for joint_idx in [5, 6]:
            change = solution[joint_idx] - initial_guess[joint_idx]
            if abs(change) > max_wrist_change:
                limited_solution[joint_idx] = initial_guess[joint_idx] + np.sign(change) * max_wrist_change
                
        return limited_solution