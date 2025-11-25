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
import heapq
import mujoco
import numpy as np
from numpy.typing import NDArray

from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    DEFAULT_MUJOCO_MODEL_PATH, DEFAULT_CONTROL_FREQ, N_JOINTS, EE_BODY_NAME,
    TCP_OFFSET, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER, TORQUE_LIMITS,
    DEFAULT_KP_REMOTE, DEFAULT_KD_REMOTE
)

class RemoteRobotSimulator:
    def __init__(self, delay_config: ExperimentConfig = ExperimentConfig.LOW_DELAY, seed: Optional[int] = None):
        self.model_path = DEFAULT_MUJOCO_MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        
        # Separate Data: One for Physics, One for Math
        self.data = mujoco.MjData(self.model)
        self.data_control = mujoco.MjData(self.model)

        self.n_joints = N_JOINTS
        self.ee_body_name = EE_BODY_NAME
        self.control_freq = DEFAULT_CONTROL_FREQ
        
        sim_freq = int(1.0 / self.model.opt.timestep)
        self.n_substeps = sim_freq // self.control_freq

        self.torque_limits = TORQUE_LIMITS.copy()
        self.joint_limits_lower = JOINT_LIMITS_LOWER.copy()
        self.joint_limits_upper = JOINT_LIMITS_UPPER.copy()
        self.tcp_offset = TCP_OFFSET.copy()
        self.kp = DEFAULT_KP_REMOTE
        self.kd = DEFAULT_KD_REMOTE

        self.last_q_target = np.zeros(self.n_joints)
        self.delay_simulator = DelaySimulator(self.control_freq, config=delay_config, seed=seed)
        self.action_queue: List[Tuple[int, np.ndarray]] = []
        self.internal_tick = 0
        self.last_executed_rl_torque = np.zeros(self.n_joints)

    def _reset_mujoco_data(self, mj_data: mujoco.MjData, q_init: NDArray[np.float64]) -> None:
        """
        Reset a MuJoCo data structure to a specific joint configuration.
        
        CRITICAL: Does NOT use mj_resetData because that sets qpos to XML defaults
        first, which can cause visualization glitches and incorrect initial states.
        """
        # Clear all velocities
        mj_data.qvel[:] = 0.0
        
        # Clear all accelerations
        mj_data.qacc[:] = 0.0
        mj_data.qacc_warmstart[:] = 0.0
        
        # Clear all controls
        mj_data.ctrl[:] = 0.0
        
        # Clear applied forces
        mj_data.qfrc_applied[:] = 0.0
        mj_data.xfrc_applied[:] = 0.0
        
        # Reset time
        mj_data.time = 0.0
        
        # Set joint positions to our initial config
        mj_data.qpos[:self.n_joints] = q_init
        
        # Compute forward kinematics
        mujoco.mj_forward(self.model, mj_data)
        
    def reset(self, initial_qpos: NDArray[np.float64]) -> None:
        """
        Reset simulation and stabilize physics at the exact initial config.
        
        CRITICAL: Does NOT use mj_resetData to avoid MuJoCo XML defaults.
        """
        # 1. Reset internal state
        self.action_queue = []
        self.internal_tick = 0
        self.last_executed_rl_torque = np.zeros(self.n_joints)
        self.last_q_target = initial_qpos.copy()
        
        # 2. Reset PHYSICS data (NO mj_resetData!)
        self._reset_mujoco_data(self.data, initial_qpos)
        
        # 3. Reset CONTROL data for inverse dynamics
        self._reset_mujoco_data(self.data_control, initial_qpos)

        # 4. Calculate Gravity Compensation for this static pose
        self.data_control.qpos[:self.n_joints] = initial_qpos
        self.data_control.qvel[:self.n_joints] = 0.0
        self.data_control.qacc[:self.n_joints] = 0.0
        mujoco.mj_inverse(self.model, self.data_control)
        gravity_torque = self.data_control.qfrc_inverse[:self.n_joints].copy()
        
        # 5. Settling Loop (Anti-Shock)
        # Run 100 steps holding the position perfectly still to let contacts settle
        for _ in range(100):
            self.data.ctrl[:self.n_joints] = gravity_torque
            self.data.qvel[:self.n_joints] = 0.0  # Kill kinetic energy
            mujoco.mj_step(self.model, self.data)

    def _normalize_angle(self, angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _get_inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a_desired: np.ndarray) -> np.ndarray:
        # Use separate control data structure to avoid corrupting physics
        self.data_control.qpos[:self.n_joints] = q
        self.data_control.qvel[:self.n_joints] = v
        self.data_control.qacc[:self.n_joints] = a_desired
        mujoco.mj_inverse(self.model, self.data_control)
        return self.data_control.qfrc_inverse[:self.n_joints].copy()
    
    def step(self, target_q, target_qd, torque_compensation) -> dict:
        self.internal_tick += 1
        
        delay_steps = int(self.delay_simulator.get_action_delay_steps())
        arrival_time = self.internal_tick + delay_steps
        heapq.heappush(self.action_queue, (arrival_time, torque_compensation.copy()))
        
        while self.action_queue and self.action_queue[0][0] <= self.internal_tick:
            _, self.last_executed_rl_torque = heapq.heappop(self.action_queue)
            
        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()
        
        q_error = self._normalize_angle(target_q - q_current) 
        qd_error = target_qd - qd_current
        acc_desired = self.kp * q_error + self.kd * qd_error 
        
        tau_id = self._get_inverse_dynamics(q_current, qd_current, acc_desired)
        tau_id[-1] = 0.0
        self.last_executed_rl_torque[-1] = 0.0

        tau_total = tau_id + self.last_executed_rl_torque
        tau_clipped = np.clip(tau_total, -self.torque_limits, self.torque_limits)
        limits_hit = np.any(tau_total != tau_clipped)
        
        self.data.ctrl[:self.n_joints] = tau_clipped
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        return {
            "joint_error": np.linalg.norm(target_q - self.data.qpos[:self.n_joints]),
            "tau_pd": tau_id, "tau_rl": self.last_executed_rl_torque, "tau_total": tau_total
        }

    def get_joint_state(self):
        return self.data.qpos[:self.n_joints].copy(), self.data.qvel[:self.n_joints].copy()