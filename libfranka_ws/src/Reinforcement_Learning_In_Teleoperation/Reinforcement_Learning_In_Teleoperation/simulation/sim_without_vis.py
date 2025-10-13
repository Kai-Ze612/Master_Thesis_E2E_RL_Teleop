# robot_control/visualization/mujoco_viewer.py
# robot_control/simulation/robot_simulator.py
"""
Core robot simulator - physics only, no visualization.
Can run headless for fast training.
"""

from __future__ import annotations
from typing import Optional

import mujoco
import numpy as np
from numpy.typing import NDArray

from robot_control.controllers.inverse_dynamics_controller import InverseDynamicsController
from robot_control.dynamics.mujoco_dynamics import MuJoCoDynamics
from robot_control.utils.kinematics import InverseKinematicsSolver


class RobotSimulator:
    """Core robot simulator without visualization.
    
    This class handles:
    - Physics simulation
    - Inverse kinematics
    - Control computation
    - State queries
    
    Does NOT handle:
    - Rendering (separate class)
    - Reward computation (environment wrapper)
    - Episode management (environment wrapper)
    """
    
    _N_JOINTS = 7
    _EE_BODY_NAME = 'panda_hand'
    _TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float64)
    
    def __init__(
        self,
        model_path: str,
        control_freq: int,
        torque_limits: NDArray[np.float64],
        joint_limits_lower: NDArray[np.float64],
        joint_limits_upper: NDArray[np.float64],
    ) -> None:
        """Initialize robot simulator."""
        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Time
        self._dt = 1.0 / control_freq
        self._control_freq = control_freq
        
        # Compute substeps
        sim_freq = int(1.0 / self.model.opt.timestep)
        if sim_freq % control_freq != 0:
            raise ValueError(
                f"Sim freq ({sim_freq}) must be multiple of control freq ({control_freq})"
            )
        self._n_substeps = sim_freq // control_freq
        
        # Limits
        self._torque_limits = torque_limits.copy()
        
        # IK solver
        self._ik_solver = InverseKinematicsSolver(
            self.model, joint_limits_lower, joint_limits_upper
        )
        
        # Controller
        dynamics = MuJoCoDynamics(self.model, self.data, self._N_JOINTS)
        self._controller = InverseDynamicsController(dynamics)
        
        # State tracking
        self._current_time = 0.0
        self._last_q_target = np.zeros(self._N_JOINTS, dtype=np.float64)
    
    @property
    def dt(self) -> float:
        """Control timestep."""
        return self._dt
    
    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return self._N_JOINTS
    
    def reset(self, initial_qpos: NDArray[np.float64]) -> None:
        """Reset simulation."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self._N_JOINTS] = initial_qpos
        self.data.qvel[:self._N_JOINTS] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        self._current_time = 0.0
        self._last_q_target = initial_qpos.copy()
    
    def step(
        self,
        target_pos: NDArray[np.float64],
        rl_action: NDArray[np.float64]
    ) -> None:
        """Execute one control step."""
        self._current_time += self._dt
        
        # Current state
        q = self.data.qpos[:self._N_JOINTS].copy()
        qd = self.data.qvel[:self._N_JOINTS].copy()
        
        # IK
        q_target, success = self._ik_solver.solver(target_pos, q, self._EE_BODY_NAME)
        if not success:
            q_target = self._last_q_target
        self._last_q_target = q_target.copy()
        
        # Target velocity
        qd_target = (q_target - self._last_q_target) / self._dt
        
        # Baseline control
        tau_baseline = self._controller.compute_control(q, qd, q_target, qd_target)
        
        # RL correction
        action_clipped = np.clip(rl_action, -0.5, 0.5)
        tau_total = tau_baseline * (1.0 + action_clipped)
        
        # Apply limits
        tau_clipped = np.clip(tau_total, -self._torque_limits, self._torque_limits)
        
        # Send to MuJoCo
        self.data.ctrl[:self._N_JOINTS] = tau_clipped
        
        # Simulate (no rendering here!)
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)
    
    def get_state(self) -> dict[str, NDArray[np.float64]]:
        """Get robot state."""
        return {
            "joint_pos": self.data.qpos[:self._N_JOINTS].copy(),
            "joint_vel": self.data.qvel[:self._N_JOINTS].copy(),
        }
    
    def get_ee_position(self) -> NDArray[np.float64]:
        """Get end-effector position."""
        ee_id = self.model.body(self._EE_BODY_NAME).id
        flange_pos = self.data.xpos[ee_id].copy()
        flange_rot = self.data.xmat[ee_id].reshape(3, 3)
        tcp_pos = flange_pos + flange_rot @ self._TCP_OFFSET
        return tcp_pos