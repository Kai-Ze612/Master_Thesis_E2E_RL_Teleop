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
from typing import Optional, Tuple

import mujoco
import numpy as np
from numpy.typing import NDArray

from Reinforcement_Learning_In_Teleoperation.controllers.inverse_kinematics import IKSolver
from Reinforcement_Learning_In_Teleoperation.controllers.pd_controller import AdaptivePDController


class RemoteRobotSimulator:
    def __init__(
        self,
        model_path: str,
        control_freq: int,
        torque_limits: NDArray[np.float64],
        joint_limits_lower: NDArray[np.float64],
        joint_limits_upper: NDArray[np.float64],
        # Adaptive PD parameters
        kp_nominal: Optional[NDArray[np.float64]] = None,
        kd_nominal: Optional[NDArray[np.float64]] = None,
        min_gain_ratio: float = 0.3,
        delay_threshold: float = 0.2,
        # IK solver parameters
        jacobian_max_iter: int = 100,
        position_tolerance: float = 1e-4,
        jacobian_step_size: float = 0.25,
        jacobian_damping: float = 1e-4,
        max_joint_change: float = 0.1,
        continuity_gain: float = 0.5,
    ):
        
        # Initialize MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_joints = 7
        self.ee_body_name = "panda_hand"

        # Time step configuration
        self.dt = 1.0 / control_freq
        self.control_freq = control_freq

        # Simulation frequency and substeps
        sim_freq = int(1.0 / self.model.opt.timestep)
        if sim_freq % control_freq != 0:
            raise ValueError(f"Simulation frequency ({sim_freq} Hz) must be a multiple of control frequency ({control_freq} Hz).")

        self.n_substeps = sim_freq // control_freq

        # Actuator and joint limits
        self.torque_limits = torque_limits
        self.joint_limits_lower = joint_limits_lower
        self.joint_limits_upper = joint_limits_upper

        # TCP offset from flange to end-effector (in meters)
        self.tcp_offset = np.array([0.0, 0.0, 0.1034])

        # Initialize IK solver and PD controller
        self.ik_solver = IKSolver(
            model=self.model,
            joint_limits_lower=joint_limits_lower,
            joint_limits_upper=joint_limits_upper,
            jacobian_max_iter=jacobian_max_iter,
            position_tolerance=position_tolerance,
            jacobian_step_size=jacobian_step_size,
            jacobian_damping=jacobian_damping,
            optimization_max_iter=100,
            optimization_ftol=1e-8,
            optimization_xtol=1e-8,
            max_joint_change=max_joint_change,
            continuity_gain=continuity_gain,
        )

        self.pd_controller = AdaptivePDController(
            n_joints=self.n_joints,
            kp_nominal=kp_nominal,
            kd_nominal=kd_nominal,
            min_gain_ratio=min_gain_ratio,
            delay_threshold=delay_threshold,
        )

        # State tracking for velocity estimation
        self.last_q_target = np.zeros(self.n_joints)
        self.last_timestamp = 0.0
        
        # Debug information storage
        self.debug_info = {}

    def reset(
        self, 
        initial_qpos: NDArray[np.float64],
        reset_controllers: bool = True
    ) -> None:
        """ Reset the simulation to the initial joint configuration."""
        
        if initial_qpos.shape != (self.n_joints,):
            raise ValueError(
                f"initial_qpos must have shape ({self.n_joints},), "
                f"got {initial_qpos.shape}"
            )

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = initial_qpos
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)

        # Reset controller states
        self.last_q_target = initial_qpos.copy()
        self.last_timestamp = 0.0
        
        if reset_controllers:
            # Reset IK solver's trajectory tracking
            self.ik_solver.reset_trajectory(q_start=initial_qpos)
            
            # Reset PD controller gains to nominal (zero delay)
            self.pd_controller.update_gains(delay=0.0)

        self.debug_info = {}

    def compute_inverse_dynamics_torque(
        self,
        q: NDArray[np.float64],
        qd: NDArray[np.float64],
        qdd: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Inverse dynamics using MuJoCo's built-in function."""
        
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
        tau = self.data.qfrc_inverse[:self.n_joints].copy()

        # Restore original state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.qacc[:] = qacc_save
        mujoco.mj_forward(self.model, self.data)

        return tau

    def step(
        self,
        target_pos: NDArray[np.float64],
        normalized_action: NDArray[np.float64],
        current_delay: Optional[float] = None,
        target_vel_cartesian: Optional[NDArray[np.float64]] = None,
    ) -> dict:
        
        # Checking up input position
        if target_pos.shape != (3,):
            raise ValueError(f"target_pos must have shape (3,), got {target_pos.shape}")
        if normalized_action.shape != (self.n_joints,):
            raise ValueError(
                f"normalized_action must have shape ({self.n_joints},), "
                f"got {normalized_action.shape}"
            )

        # Update PD gains based on current delay measurement
        if current_delay is not None:
            self.pd_controller.update_gains(delay=current_delay)

        # Get current Robot state
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()

        # IK
        q_target, ik_success, ik_error = self.ik_solver.solve(
            target_pos=target_pos,
            q_init=q_current,
            body_name=self.ee_body_name,
            tcp_offset=self.tcp_offset,
            enforce_continuity=True,  # Use trajectory continuity enforcement
        )

        if q_target is None or not ik_success:
            print(
                f"Warning: IK failed with error {ik_error:.6f}m. "
                f"Using previous target."
            )
            q_target = self.last_q_target.copy()

        
        # Method 1: Finite difference of joint-space targets (always available)
        q_target_old = self.last_q_target.copy()
        qd_target_fd = (q_target - q_target_old) / self.dt

        # Method 2: If Cartesian velocity provided, use Jacobian mapping
        if target_vel_cartesian is not None:
            # Compute Jacobian at current configuration
            jacp = np.zeros((3, self.model.nv))
            ee_id = self.model.body(self.ee_body_name).id
            ee_pos = self.get_ee_position()
            
            mujoco.mj_jac(
                self.model,
                self.data,
                jacp,
                None,  # We only need position Jacobian
                ee_pos,
                ee_id
            )
            
            J = jacp[:, :self.n_joints]
            
            # Pseudo-inverse mapping: q̇ = J⁺·v_cartesian
            try:
                J_pinv = np.linalg.pinv(J)
                qd_target_jacobian = J_pinv @ target_vel_cartesian
                
                # Blend with finite difference (safety check)
                # If Jacobian-based velocity is too different, trust FD more
                velocity_discrepancy = np.linalg.norm(qd_target_jacobian - qd_target_fd)
                if velocity_discrepancy < 1.0:  # Reasonable agreement
                    qd_target = qd_target_jacobian
                else:
                    print(f"Warning: Large velocity discrepancy {velocity_discrepancy:.3f}, using FD")
                    qd_target = qd_target_fd
            except np.linalg.LinAlgError:
                qd_target = qd_target_fd
        else:
            qd_target = qd_target_fd

        # Update state tracking
        self.last_q_target = q_target.copy()

        # ============================================================
        # Adaptive PD Control: Compute Desired Acceleration
        # ============================================================
        qdd_desired = self.pd_controller.compute_desired_acceleration(
            q_current=q_current,
            qd_current=qd_current,
            q_target=q_target,
            qd_target=qd_target,
            delay=None,  # Already updated gains above if delay provided
        )

        # ============================================================
        # Inverse Dynamics: Compute Baseline Torque
        # ============================================================
        tau_baseline = self.compute_inverse_dynamics_torque(
            q=q_current,
            qd=qd_current,
            qdd=qdd_desired,
        )

        # ============================================================
        # RL Torque Compensation (Multiplicative)
        # ============================================================
        action_clipped = np.clip(normalized_action, -0.5, 0.5)
        tau_total = tau_baseline * (1.0 + action_clipped)

        # ============================================================
        # Apply Actuator Limits
        # ============================================================
        tau_clipped = np.clip(tau_total, -self.torque_limits, self.torque_limits)

        # Check if limits were hit (for debugging)
        limits_hit = np.any(
            (tau_total < -self.torque_limits) | (tau_total > self.torque_limits)
        )

        # ============================================================
        # Execute Control Command
        # ============================================================
        self.data.ctrl[:self.n_joints] = tau_clipped

        # Simulate forward for one control step (multiple physics substeps)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # ============================================================
        # Compute Performance Metrics
        # ============================================================
        
        # Position tracking error (Cartesian space)
        ee_pos_achieved = self.get_ee_position()
        position_error = np.linalg.norm(target_pos - ee_pos_achieved)

        # Joint space tracking error
        q_achieved = self.data.qpos[:self.n_joints].copy()
        joint_position_error = np.linalg.norm(q_target - q_achieved)

        # Velocity tracking error
        qd_achieved = self.data.qvel[:self.n_joints].copy()
        joint_velocity_error = np.linalg.norm(qd_target - qd_achieved)

        # Get current controller gains
        kp_current, kd_current = self.pd_controller.get_current_gains()

        # ============================================================
        # Store Debug Information
        # ============================================================
        self.debug_info = {
            # Torque decomposition
            "tau_baseline": tau_baseline,
            "tau_rl_correction": tau_baseline * action_clipped,
            "action_clipped": action_clipped,
            "tau_total": tau_total,
            "tau_clipped": tau_clipped,
            "limits_hit": limits_hit,
            
            # IK information
            "ik_success": ik_success,
            "ik_error": ik_error,
            "q_target": q_target,
            
            # Tracking errors
            "position_error_cartesian": position_error,
            "position_error_joint": joint_position_error,
            "velocity_error_joint": joint_velocity_error,
            
            # Controller state
            "current_delay": current_delay if current_delay is not None else self.pd_controller.current_delay,
            "gain_ratio": self.pd_controller.current_gain_ratio,
            "kp_current": kp_current,
            "kd_current": kd_current,
            
            # Desired vs achieved
            "qdd_desired": qdd_desired,
            "q_achieved": q_achieved,
            "qd_achieved": qd_achieved,
        }

        # ============================================================
        # Return Step Information
        # ============================================================
        step_info = {
            "position_error": position_error,
            "joint_error": joint_position_error,
            "velocity_error": joint_velocity_error,
            "ik_success": ik_success,
            "limits_hit": limits_hit,
            "current_delay": self.debug_info["current_delay"],
            "gain_ratio": self.pd_controller.current_gain_ratio,
        }

        return step_info

    def get_state(self) -> dict:
        """
        Returns the current state of the follower robot.
        
        Returns:
            dict with keys:
                - joint_pos: Current joint positions (7,)
                - joint_vel: Current joint velocities (7,)
                - ee_pos: End-effector Cartesian position (3,)
                - gravity_torque: Gravity compensation torques (7,)
                - controller_gains: Current adaptive PD gains
        """
        q = self.data.qpos[:self.n_joints].copy()
        qd = self.data.qvel[:self.n_joints].copy()
        ee_pos = self.get_ee_position()

        # Compute gravity torque by setting velocities to zero
        qvel_save = self.data.qvel.copy()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        G = -self.data.qfrc_bias[:self.n_joints].copy()
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)

        # Get current controller gains
        kp_current, kd_current = self.pd_controller.get_current_gains()

        return {
            "joint_pos": q,
            "joint_vel": qd,
            "ee_pos": ee_pos,
            "gravity_torque": G,
            "controller_gains": {
                "kp": kp_current,
                "kd": kd_current,
                "gain_ratio": self.pd_controller.current_gain_ratio,
                "current_delay": self.pd_controller.current_delay,
            },
        }

    def get_ee_position(self) -> NDArray[np.float64]:
        """
        Return the current end-effector (TCP) position in world coordinates.
        
        Returns:
            tcp_position: 3D position of TCP in world frame (3,)
        """
        ee_id = self.model.body(self.ee_body_name).id
        flange_position = self.data.xpos[ee_id].copy()
        flange_rotation = self.data.xmat[ee_id].reshape(3, 3)
        tcp_position = flange_position + flange_rotation @ self.tcp_offset
        return tcp_position

    def get_debug_info(self) -> dict:
        """
        Get comprehensive debug information about the control system.
        
        Returns:
            dict containing torque decomposition, tracking errors, and controller state
        """
        return self.debug_info.copy()

    def set_nominal_gains(
        self,
        kp_nominal: Optional[NDArray[np.float64]] = None,
        kd_nominal: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Update nominal PD gains during runtime.
        
        Useful for online tuning or different task requirements.
        
        Args:
            kp_nominal: New nominal proportional gains
            kd_nominal: New nominal derivative gains
        """
        self.pd_controller.set_nominal_gains(kp_nominal, kd_nominal)

    def get_controller_info(self) -> dict:
        """
        Get information about controller configuration.
        
        Returns:
            dict with controller parameters and current state
        """
        kp_nom, kd_nom = self.pd_controller.get_nominal_gains()
        kp_cur, kd_cur = self.pd_controller.get_current_gains()

        return {
            "nominal_gains": {"kp": kp_nom, "kd": kd_nom},
            "current_gains": {"kp": kp_cur, "kd": kd_cur},
            "gain_adaptation": {
                "min_gain_ratio": self.pd_controller.min_gain_ratio,
                "delay_threshold": self.pd_controller.delay_threshold,
                "current_ratio": self.pd_controller.current_gain_ratio,
            },
            "ik_solver_config": {
                "position_tolerance": self.ik_solver.position_tolerance,
                "max_joint_change": self.ik_solver.max_joint_change,
                "continuity_gain": self.ik_solver.continuity_gain,
            },
        }