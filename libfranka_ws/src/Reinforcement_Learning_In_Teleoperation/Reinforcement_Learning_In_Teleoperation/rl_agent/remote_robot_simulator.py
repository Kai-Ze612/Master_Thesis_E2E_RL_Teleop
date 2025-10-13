"""
MuJoCo-based simulator for the remote robot (follower).

This module implements:
- Inverse kinematics to convert target end-effector positions to joint angles.
- Adaptive PD control, when delay is high, PD gains are decreased, when delay is low, PD gains are increased.
- We use MuJoco's built-in inverse dynamics function to compute the baseline torque.
- RL agent learns torque compensation
- Interpolation using NN predictor to predict missing trajectory points due to delay. 
"""

from __future__ import annotations
from typing import Optional

import mujoco
import numpy as np
from numpy.typing import NDArray

from Reinforcement_Learning_In_Teleoperation.controllers.inverse_kinematics import IKSolver

class RemoteRobotSimulator:
    def __init__(self, 
                 model_path: str, 
                 control_freq: int,
                 torque_limits: np.ndarray, 
                 joint_limits_lower: np.ndarray,
                 joint_limits_upper: np.ndarray):
        """
        Initializes the remote robot simulator.
        
        Args:
            model_path: Path to MuJoCo XML model file
            control_freq: Control frequency in Hz
            torque_limits: Joint torque limits (N⋅m)
            joint_limits_lower: Lower joint position limits (rad)
            joint_limits_upper: Upper joint position limits (rad)
        """

        # Initialize MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_joints = 7
        self.ee_body_name = 'panda_hand'

        # Initialize IK solver
        self.ik_solver = IKSolver(
            model=self.model,
            joint_limits_lower=joint_limits_lower,
            joint_limits_upper=joint_limits_upper
        )
        
        # Time step
        self.dt = 1.0 / control_freq
        self.control_freq = control_freq

        # Simulation frequency and substeps
        sim_freq = int(1.0 / self.model.opt.timestep)
        if sim_freq % control_freq != 0:
            raise ValueError(
                f"Simulation frequency ({sim_freq} Hz) must be a multiple "
                f"of control frequency ({control_freq} Hz)."
            )
        self.n_substeps = sim_freq // control_freq

        # Controller parameters
        self.torque_limits = torque_limits
        self.joint_limits_lower = joint_limits_lower
        self.joint_limits_upper = joint_limits_upper

        # TCP offset from flange to end-effector (in meters)
        self.tcp_offset = np.array([0.0, 0.0, 0.1034])
        
        # PD gains for computing desired acceleration
        self.kp = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
        self.kd = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])
        
        # State variables
        self.last_q_target = np.zeros(self.n_joints)

    def reset(self, initial_qpos: np.ndarray) -> None:
        """
        Resets the robot to an initial joint configuration.
        
        Args:
            initial_qpos: Initial joint positions (7,)
        """
        if initial_qpos.shape != (self.n_joints,):
            raise ValueError(
                f"initial_qpos must have shape ({self.n_joints},), "
                f"got {initial_qpos.shape}"
            )
        
        mujoco.mj_resetData(self.model, self.data)
        
        self.data.qpos[:self.n_joints] = initial_qpos
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.last_q_target = initial_qpos.copy()

    def compute_inverse_dynamics_torque(self,
                                       q: np.ndarray,
                                       qd: np.ndarray,
                                       qdd: np.ndarray) -> np.ndarray:
        """
        Compute inverse dynamics torque using MuJoCo's built-in function.
        
        Uses mj_inverse() which computes: τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
        
        Args:
            q: Joint positions (7,)
            qd: Joint velocities (7,)
            qdd: Desired joint accelerations (7,)
            
        Returns:
            tau: Required joint torques (7,)
        """
        # Save current state
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        qacc_save = self.data.qacc.copy()
        
        # Set desired state
        self.data.qpos[:self.n_joints] = q
        self.data.qvel[:self.n_joints] = qd
        self.data.qacc[:self.n_joints] = qdd
        
        # Compute inverse dynamics using MuJoCo's built-in function
        mujoco.mj_inverse(self.model, self.data)
        
        # Get computed torques
        tau = self.data.qfrc_inverse[:self.n_joints].copy()
        
        # Restore original state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.qacc[:] = qacc_save
        mujoco.mj_forward(self.model, self.data)
        
        return tau
    
    def step(self, target_pos: np.ndarray, normalized_action: np.ndarray) -> None:
        """
        Execute one control step.
        
        Args:
            target_pos: Target end-effector position (3,) - potentially delayed or NN-predicted
            normalized_action: RL agent's torque correction in range [-0.5, 0.5] (7,)
        
        Note: This method expects target_pos to already be compensated for delay
              (either by NN predictor or left as delayed observation for RL to handle).
        """
        # Validate inputs
        if target_pos.shape != (3,):
            raise ValueError(f"target_pos must have shape (3,), got {target_pos.shape}")
        if normalized_action.shape != (self.n_joints,):
            raise ValueError(
                f"normalized_action must have shape ({self.n_joints},), "
                f"got {normalized_action.shape}"
            )
        
        # Get current robot state
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        
        # Save old target for velocity computation
        q_target_old = self.last_q_target.copy()
        
        # Solve inverse kinematics
        q_target, ik_success, ik_error = self.ik_solver.solve(
            target_pos=target_pos,
            q_init=q_current,
            body_name=self.ee_body_name,
            tcp_offset=self.tcp_offset
        )

        # Handle IK failure - use previous target
        if q_target is None or not ik_success:
            q_target = self.last_q_target.copy()
            
        self.last_q_target = q_target.copy()
        
        # Estimate target velocity using finite differences
        qd_target = (q_target - q_target_old) / self.dt
        
        # Compute desired acceleration using PD control law
        e_pos = q_target - q_current
        e_vel = qd_target - qd_current
        qdd_desired = self.kp * e_pos + self.kd * e_vel
        
        # Compute baseline torque using inverse dynamics
        tau_baseline = self.compute_inverse_dynamics_torque(
            q=q_current,
            qd=qd_current,
            qdd=qdd_desired
        )
        
        # Apply RL multiplicative correction
        action_clipped = np.clip(normalized_action, -0.5, 0.5)
        tau_total = tau_baseline * (1.0 + action_clipped)
        
        # Clip to actuator limits
        tau_clipped = np.clip(tau_total, -self.torque_limits, self.torque_limits)
        
        # Apply torque command
        self.data.ctrl[:self.n_joints] = tau_clipped
        
        # Simulate forward for one control step
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Store debug information
        self.debug_info = {
            'tau_baseline': tau_baseline,
            'action_clipped': action_clipped,
            'tau_total': tau_total,
            'tau_clipped': tau_clipped,
            'ik_success': ik_success,
            'ik_error': ik_error,
        }
            
    def get_state(self) -> dict:
        """
        Returns the current state of the follower robot.
        
        Returns:
            dict with keys:
                - joint_pos: Current joint positions (7,)
                - joint_vel: Current joint velocities (7,)
                - gravity_torque: Gravity compensation torques (7,)
        """
        q = self.data.qpos[:self.n_joints].copy()
        qd = self.data.qvel[:self.n_joints].copy()
        
        # Compute gravity torque by setting velocities to zero
        qvel_save = self.data.qvel.copy()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        G = -self.data.qfrc_bias[:self.n_joints].copy()
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)
        
        return {
            "joint_pos": q,
            "joint_vel": qd,
            "gravity_torque": G,
        }
        
    def get_ee_position(self) -> np.ndarray:
        """
        Return the current end-effector (TCP) position in world coordinates.
        
        Returns:
            tcp_position: 3D position of TCP in world frame (3,)
        """
        # Get the position of the flange from MuJoCo
        ee_id = self.model.body(self.ee_body_name).id
        flange_position = self.data.xpos[ee_id].copy()
        
        # Get rotation matrix for proper TCP offset transformation
        flange_rotation = self.data.xmat[ee_id].reshape(3, 3)
        
        # Transform TCP offset from flange frame to world frame
        tcp_position = flange_position + flange_rotation @ self.tcp_offset
        return tcp_position

    def get_debug_info(self) -> dict:
        """
        Get debug information about torque decomposition.
        
        Returns:
            dict with keys:
                - tau_baseline: Baseline inverse dynamics torque
                - action_clipped: Clipped RL action
                - tau_total: Total torque before actuator limits
                - tau_clipped: Final applied torque
                - ik_success: Whether IK succeeded
                - ik_error: IK position error
        """
        return getattr(self, 'debug_info', {})