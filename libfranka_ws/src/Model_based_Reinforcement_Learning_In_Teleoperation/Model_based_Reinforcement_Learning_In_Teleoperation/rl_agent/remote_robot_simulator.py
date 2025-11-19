"""
MuJoCo-based simulator for the remote robot (follower).

Pipelines:
1. subscribe to predicted local robot state
2. subscribe to torque compensation from RL
3. PD control with inverse dynamics to compute required torques
4. final tau = baseline tau + torque compensation
5. step the MuJoCo simulation
"""

from __future__ import annotations
from cmath import tau
from typing import Tuple, Optional
from collections import deque

import mujoco
import numpy as np
from numpy.typing import NDArray

from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    DEFAULT_MUJOCO_MODEL_PATH,
    DEFAULT_CONTROL_FREQ,
    N_JOINTS,
    EE_BODY_NAME,
    TCP_OFFSET,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    TORQUE_LIMITS,
    DEFAULT_KP_REMOTE,
    DEFAULT_KD_REMOTE,
)

class RemoteRobotSimulator:
    """MuJoCo-based simulator for the remote robot (follower)."""
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.LOW_DELAY,
        seed: Optional[int] = None,
    ):
        # Initialize MuJoCo model and data
        self.model_path = DEFAULT_MUJOCO_MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Robot configuration
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
        self.kp = DEFAULT_KP_REMOTE
        self.kd = DEFAULT_KD_REMOTE

        # State tracking
        self.last_q_target = np.zeros(self.n_joints)

        # Action delay
        temp_delay_sim = DelaySimulator(self.control_freq, config=delay_config, seed=seed)
        self.action_delay_steps = temp_delay_sim.get_action_delay_steps()
        
        # Create a FIFO buffer for torque actions
        # If delay is 0, maxlen=1 implies immediate use (requires slight logic adjustment below)
        self.torque_buffer = deque(maxlen=max(1, self.action_delay_steps))
        
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

        self.torque_buffer.clear()
        
        if self.action_delay_steps > 0:
            for _ in range(self.action_delay_steps):
                self.torque_buffer.append(np.zeros(self.n_joints))
        
    def _normalize_angle(self, angle: np.ndarray) -> np.ndarray:
        """Normalize an angle or array of angles to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    # def compute_gravity_compensation(
    #     self,
    #     q: NDArray[np.float64],
    # ) -> NDArray[np.float64]:
    #     """Gravity compensation torque for given joint positions."""
    
    #     # Save current state
    #     qpos_save = self.data.qpos.copy()
    #     qvel_save = self.data.qvel.copy()
    #     qacc_save = self.data.qacc.copy()

    #     # Set desired state
    #     self.data.qpos[:self.n_joints] = q
    #     self.data.qvel[:self.n_joints] = 0.0
    #     self.data.qacc[:self.n_joints] = 0.0

    #     mujoco.mj_inverse(self.model, self.data)
    #     tau_gravity = self.data.qfrc_inverse[:self.n_joints].copy()

    #     # Restore original state
    #     self.data.qpos[:] = qpos_save
    #     self.data.qvel[:] = qvel_save
    #     self.data.qacc[:] = qacc_save
        
    #     # Reset the data state after restoring
    #     mujoco.mj_forward(self.model, self.data)

    #     return tau_gravity

    def step(
        self,
        target_q: NDArray[np.float64],
        torque_compensation: NDArray[np.float64],
    ) -> dict:
        """Execute one control step with additive torque compensation."""
        
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        q_target = target_q
       
        # Calculate the target velocity
        q_target_old = self.last_q_target.copy()
        qd_target = (q_target - q_target_old) / self.dt
        self.last_q_target = q_target.copy()

        # Calculate desired acceleration using PD control
        q_error = self._normalize_angle(q_target - q_current) # Normalize angle error
        qd_error = qd_target - qd_current
        qdd_desired = self.kp * q_error + self.kd * qd_error

        # Compute baseline torque using inverse dynamics
        # tau_gravity = self.compute_gravity_compensation(q_current)
        tau_pd = self.kp * q_error + self.kd * qd_error
        # tau_baseline = tau_gravity + tau_pd
       
        # Applying RL compensation
        tau_total = tau_pd + torque_compensation

        if self.action_delay_steps > 0:
            torque_to_apply = self.torque_buffer[0]
            self.torque_buffer.append(tau_total.copy())
        else:
            # Immediate execution if no delay
            torque_to_apply = tau_total
        
        # Apply safety torque limits
        tau_clipped = np.clip(torque_to_apply, -self.torque_limits, self.torque_limits)
        limits_hit = np.any(torque_to_apply != tau_clipped)
        
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