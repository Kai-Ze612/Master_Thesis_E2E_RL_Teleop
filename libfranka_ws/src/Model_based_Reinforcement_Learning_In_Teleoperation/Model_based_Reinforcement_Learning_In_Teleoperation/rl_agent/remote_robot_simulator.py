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
from typing import Tuple, Optional, List
import heapq  # For time order queue management

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

        # Adding action delay simulator
        self.delay_simulator = DelaySimulator(self.control_freq, config=delay_config, seed=seed)
        self.action_queue: List[Tuple[int, np.ndarray]] = []
        
        self.internal_tick = 0
        self.last_executed_rl_torque = np.zeros(self.n_joints)
        
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

        # Reset queue and time
        self.action_queue = []
        self.internal_tick = 0
        self.last_executed_rl_torque = np.zeros(self.n_joints)
        
    def _normalize_angle(self, angle: np.ndarray) -> np.ndarray:
        """Normalize an angle or array of angles to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _get_inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a_desired: np.ndarray) -> np.ndarray:
        """
        Compute Torque using MuJoCo Inverse Dynamics.
        Equation: tau = M(q)*a_desired + C(q,v)*v + G(q)
        """
        # 1. Update MuJoCo with the CURRENT Robot State
        self.data.qpos[:self.n_joints] = q
        self.data.qvel[:self.n_joints] = v
        
        # 2. Set the DESIRED Acceleration (from PD)
        self.data.qacc[:self.n_joints] = a_desired

        # 3. Compute Inverse Dynamics
        mujoco.mj_inverse(self.model, self.data)

        return self.data.qfrc_inverse[:self.n_joints].copy()
    
    def step(
        self,
        target_q: NDArray[np.float64],
        target_qd: NDArray[np.float64],
        torque_compensation: NDArray[np.float64],
    ) -> dict:
        """Execute one control step with Computed Torque Control."""
        
        self.internal_tick += 1
        
        # Adding action delay
        delay_steps = int(self.delay_simulator.get_action_delay_steps())
        arrival_time = self.internal_tick + delay_steps
        heapq.heappush(self.action_queue, (arrival_time, torque_compensation.copy()))
        
        # Hold the latest valid torque compensation that has arrived
        updated = False
        while self.action_queue and self.action_queue[0][0] <= self.internal_tick:
            _, valid_torque = heapq.heappop(self.action_queue)
            self.last_executed_rl_torque = valid_torque 
            updated = True
            
        # get remote state
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        
        # calculate acc based on PD control
        q_error = self._normalize_angle(target_q - q_current) 
        qd_error = target_qd - qd_current
        acc_desired = self.kp * q_error + self.kd * qd_error # rad/s^2
        
        # inverse dynamics (computed torque)
        tau_id = self._get_inverse_dynamics(q_current, qd_current, acc_desired)
        
        # Safety: No torque on gripper joint (index -1)
        tau_id[-1] = 0.0
        self.last_executed_rl_torque[-1] = 0.0

        # Adding noise to inverse dynamics torque for simulating model inaccuracies
        # This will help the RL agent to learn to compensate better in real world robot (due to no perfect inverse dynamics model)
        noise = np.random.uniform(0.95, 1.05, size=self.n_joints)
        tau_id = tau_id * noise
        
        # combine torques
        tau_total = tau_id + self.last_executed_rl_torque

        # Apply safety limits
        tau_clipped = np.clip(tau_total, -self.torque_limits, self.torque_limits)
        limits_hit = np.any(tau_total != tau_clipped)
        
        # step physics
        self.data.ctrl[:self.n_joints] = tau_clipped
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Metrics
        q_achieved = self.data.qpos[:self.n_joints].copy()
        joint_position_error = np.linalg.norm(target_q - q_achieved)
        qd_achieved = self.data.qvel[:self.n_joints].copy()
        joint_velocity_error = np.linalg.norm(target_qd - qd_achieved)

        step_info = {
            "joint_error": joint_position_error,
            "velocity_error": joint_velocity_error,
            "limits_hit": limits_hit,
            "tau_pd": tau_id, # Renamed for clarity (this is now the Baseline ID Torque)
            "tau_rl": self.last_executed_rl_torque,
            "tau_total": tau_total
        }
        return step_info

    def get_joint_state(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns the current joint positions and velocities."""
        return (
            self.data.qpos[:self.n_joints].copy(),
            self.data.qvel[:self.n_joints].copy()
        )