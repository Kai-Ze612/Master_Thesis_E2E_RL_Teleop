"""
This script serves as a mujoco simulator for remote robot (follower).
"""

import mujoco
import numpy as np

from Reinforcement_Learning_In_Teleoperation.controllers.inverse_kinematics import InverseKinematicsSolver

class RemoteRobotSimulator:
    def __init__(self, 
                 model_path: str, 
                 control_freq: int,
                 torque_limits: np.ndarray, 
                 joint_limits_lower: np.ndarray,
                 joint_limits_upper: np.ndarray):
        """
        Initializes the remote robot simulator.
        """

        # Initialize MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_joints = 7
        self.ee_body_name = 'panda_hand'

        # Initialize IK solver
        self.ik_solver = InverseKinematicsSolver(self.model, joint_limits_lower, joint_limits_upper)
        
        # Time step for velocity calculations
        self.dt = 1.0 / control_freq

        # Simulation frequency and substeps
        sim_freq = int(1.0 / self.model.opt.timestep)
        if sim_freq % control_freq != 0:
            raise ValueError("Simulation frequency must be a multiple of control frequency.")
        self.n_substeps = sim_freq // control_freq

        # Initialize controllers
        self.torque_limits = torque_limits
        self.joint_limits_lower = joint_limits_lower
        self.joint_limits_upper = joint_limits_upper

        # TCP offset from flange to end-effector (in meters)
        self.tcp_offset = np.array([0.0, 0.0, 0.1034])
        
        # Inverse dynamics parameters 
        self.alpha_p = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
        self.alpha_d = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])
        
        # State variables
        self.current_q_target = np.zeros(self.n_joints)
        self.last_q_target = np.zeros(self.n_joints)

    def reset(self, initial_qpos: np.ndarray):
        """Resets the robot to an initial joint configuration."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions and velocities
        self.data.qpos[:self.n_joints] = initial_qpos
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        
        # Send the simulation forward to update derived quantities
        mujoco.mj_forward(self.model, self.data)
        
        self.current_q_target = initial_qpos.copy()
        self.last_q_target = initial_qpos.copy()

    def compute_dynamics_terms(self, q: np.ndarray, qd: np.ndarray):
        """
        Compute dynamics terms: M(q), C(q,q̇)q̇, G(q)
        """
        # Save current state
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        
        # Set state to query point
        self.data.qpos[:self.n_joints] = q
        self.data.qvel[:self.n_joints] = qd
        
        # Forward kinematics to update derived quantities
        mujoco.mj_forward(self.model, self.data)
        
        # Compute M(q)
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data.qM)
        M = M_full[:self.n_joints, :self.n_joints].copy()
        
        # Compute G(q)
        qvel_temp = self.data.qvel.copy()
        qacc_temp = self.data.qacc.copy()
        
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        mujoco.mj_forward(self.model, self.data)
        
        G = -self.data.qfrc_bias[:self.n_joints].copy()
        
        # Restore velocity and acceleration
        self.data.qvel[:] = qvel_temp
        self.data.qacc[:] = qacc_temp
        mujoco.mj_forward(self.model, self.data)
        
        # Compute C(q,q̇)q̇
        C_qd = self.data.qfrc_bias[:self.n_joints].copy() - G
        
        # Restore original state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)
        
        return M, C_qd, G
    
    def compute_inverse_dynamics_torque(self, 
                                       q: np.ndarray, 
                                       qd: np.ndarray,
                                       q_target: np.ndarray,
                                       qd_target: np.ndarray = None):
        if qd_target is None:
            qd_target = np.zeros(self.n_joints)
        
        M, C_qd, G = self.compute_dynamics_terms(q, qd)
        
        # Compute position and velocity errors
        e_pos = q_target - q
        e_vel = qd_target - qd
        
        # Desired acceleration
        qdd_desired = self.alpha_p * e_pos + self.alpha_d * e_vel
        
        # Inverse dynamics
        tau_required = M @ qdd_desired + C_qd + G
        
        return tau_required
    
    def step(self, target_pos: np.ndarray, normalized_action: np.ndarray):
        
        # Get current robot state
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        
        # Convert target position to joint space using IK       
        q_target, ik_success = self.ik_solver.solver(
            target_pos,
            q_current,
            self.ee_body_name
        )

        # Handle IK failure
        if q_target is None or not ik_success:
            q_target = self.last_q_target
        else:
            self.current_q_target = q_target.copy()
            
        self.last_q_target = q_target.copy()
        
        # Estimate target velocities (finite difference)
        qd_target = (q_target - self.last_q_target) / self.dt
        
        # Using inverse dynamics to compute baseline torque
        tau_baseline = self.compute_inverse_dynamics_torque(
            q=q_current,
            qd=qd_current,
            q_target=q_target,
            qd_target=qd_target
        )
        
        # Multiplicative correction
        tau_total = tau_baseline * (1.0 + normalized_action)
        
        # Clip to actuator limits
        tau_clipped = np.clip(tau_total, -self.torque_limits, self.torque_limits)
        
        # Apply torque command
        self.data.ctrl[:self.n_joints] = tau_clipped
        
        # Simulate forward
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Store debug information
        self.debug_info = {
            'tau_baseline': tau_baseline,
            'normalized action': normalized_action,
            'tau_total': tau_total,
            'tau_clipped': tau_clipped,
        }
            
    def get_state(self) -> dict:
        """
        Returns the current state of the follower robot with dynamics info.
        """
        q = self.data.qpos[:self.n_joints].copy()
        qd = self.data.qvel[:self.n_joints].copy()
        
        # Compute dynamics for observation
        M, C_qd, G = self.compute_dynamics_terms(q, qd)
        
        return {
            "joint_pos": q,
            "joint_vel": qd,
            "gravity_torque": G,
        }
        
    def get_ee_position(self) -> np.ndarray:
        """
        Return the current end-effector (TCP) position in world coordinates.
        """
        # Get the position of the flange from MuJoCo
        ee_id = self.model.body(self.ee_body_name).id
        flange_position = self.data.xpos[ee_id].copy()
        
        # Get rotation matrix for proper TCP offset
        flange_rotation = self.data.xmat[ee_id].reshape(3, 3)
        
        # Add the offset to get the true TCP position
        tcp_position = flange_position + flange_rotation @ self.tcp_offset
        return tcp_position

    def get_debug_info(self) -> dict:
        """
        Get debug information about torque decomposition.
        """
        if hasattr(self, 'debug_info'):
            return self.debug_info
        return {}