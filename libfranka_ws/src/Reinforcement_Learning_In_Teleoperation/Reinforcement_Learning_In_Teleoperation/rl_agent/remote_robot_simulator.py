"""
MuJoCo-based simulator for the remote robot (follower).

This module implements:
- Adaptive PD control, when delay is high, PD gains are decreased, when delay is low, PD gains are increased.
- We use MuJoco's built-in inverse dynamics function to compute the baseline torque.
- RL agent learns torque compensation
- Interpolation using NN predictor to predict missing trajectory points due to delay. 

1. Torque compensation:
    inverse kinematics + RL compensation
2. Trajectory interpolation:
    NN predictor
3. Adaptive PD control:
    Adaptive PD control
"""

from __future__ import annotations
from typing import Tuple

import mujoco
import numpy as np
from numpy.typing import NDArray

from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_CONTROL_FREQ,
    N_JOINTS,
    EE_BODY_NAME,
    TCP_OFFSET,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    TORQUE_LIMITS,
    KP_REMOTE_DEFAULT,
    KD_REMOTE_DEFAULT,
)

class RemoteRobotSimulator:
    def __init__(
        self,
    ):
       
        # Initialize MuJoCo model and data
        self.model_path = DEFAULT_MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.n_joints: int = N_JOINTS
        self.ee_body_name: str = EE_BODY_NAME

        # Time step configuration
        self.control_freq: int = DEFAULT_CONTROL_FREQ
        self.dt = 1.0 / self.control_freq
        self.control_freq = self.control_freq
        
        # Simulation frequency and substeps
        sim_freq = int(1.0 / self.model.opt.timestep)
        if sim_freq % self.control_freq != 0:
            raise ValueError(f"Simulation frequency ({sim_freq} Hz) must be a multiple of control frequency ({self.control_freq} Hz).")

        self.n_substeps = sim_freq // self.control_freq

        # Actuator and joint limits
        self.torque_limits = TORQUE_LIMITS.copy()
        self.joint_limits_lower = JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = JOINT_LIMITS_UPPER.copy()

        # TCP offset from flange to end-effector (in meters)
        self.tcp_offset = TCP_OFFSET.copy()

        # PD gains
        self.kp = KP_REMOTE_DEFAULT
        self.kd = KD_REMOTE_DEFAULT

        # State tracking
        self.last_q_target = np.zeros(self.n_joints)

    def reset(
        self, 
        initial_qpos: NDArray[np.float64]
    ) -> None:
        """Reset the simulation to initial joint configuration."""
    
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = initial_qpos
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)

        self.last_q_target = initial_qpos.copy()

    def compute_inverse_dynamics_torque(
        self,
        q: NDArray[np.float64],
        qd: NDArray[np.float64],
        qdd: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Inverse dynamics using MuJoCo's built-in function.
        
        Args:
            q: Desired joint positions (7,)
            qd: Desired joint velocities (7,)
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
        tau = self.data.qfrc_inverse[:self.n_joints].copy()

        # Restore original state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.qacc[:] = qacc_save
        mujoco.mj_forward(self.model, self.data)

        return tau

    def step(
        self,
        target_q: NDArray[np.float64],
        torque_compensation: NDArray[np.float64],
    ) -> dict:
        """Execute one control step with additive torque compensation."""
        
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        q_target = target_q
       
        q_target_old = self.last_q_target.copy()
        qd_target = (q_target - q_target_old) / self.dt
        self.last_q_target = q_target.copy()

        q_error = q_target - q_current
        qd_error = qd_target - qd_current
        qdd_desired = self.kp * q_error + self.kd * qd_error

        tau_baseline = self.compute_inverse_dynamics_torque(
            q=q_current,
            qd=qd_current,
            qdd=qdd_desired,
        )
        
        tau_total = tau_baseline + torque_compensation
        tau_clipped = np.clip(tau_total, -self.torque_limits, self.torque_limits)
        limits_hit = np.any(tau_total != tau_clipped)
        
        self.data.ctrl[:self.n_joints] = tau_clipped
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        q_achieved = self.data.qpos[:self.n_joints].copy()
        joint_position_error = np.linalg.norm(q_target - q_achieved)
        qd_achieved = self.data.qvel[:self.n_joints].copy()
        joint_velocity_error = np.linalg.norm(qd_target - qd_achieved)

        step_info = {
            "joint_error": joint_position_error,
            "velocity_error": joint_velocity_error,
            "limits_hit": limits_hit,
        }
        return step_info

    def get_joint_state(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns the current joint positions and velocities."""
        return (
            self.data.qpos[:self.n_joints].copy(),
            self.data.qvel[:self.n_joints].copy()
        )