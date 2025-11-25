"""
In the experiment, we assume the leader robot is a perfect trajectory generator.

In this script, we implement a local robot simulator that generates various 2D trajectories. (all trajectories are in the XY plane with a fixed Z height for simplicity)

Current trajectory types:
- Figure-8: Smooth continuous motion, good for generalization
- Square: Sharp turns with smooth corners, tests abrupt changes
- Lissajous_Complex: A higher-order, complex, smooth trajectory, tests precision and complex tracking

The trajectory parameters can be randomized within reasonable bounds to enhance robustness during training.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import gymnasium as gym
import mujoco
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.inverse_kinematics import IKSolver
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

class TrajectoryType(Enum):
    FIGURE_8 = "figure_8"
    SQUARE = "square"
    LISSAJOUS_COMPLEX = "lissajous_complex" 


@dataclass(frozen=True) 
class TrajectoryParams:
    """
    The dataclass holds parameters for trajectory parameters
    """
    center: np.ndarray = field(default_factory=lambda: cfg.TRAJECTORY_CENTER.copy())
    scale: np.ndarray = field(default_factory=lambda: cfg.TRAJECTORY_SCALE.copy())
    frequency: float = cfg.TRAJECTORY_FREQUENCY
    initial_phase: float = 0.0


class TrajectoryGenerator(ABC):
    def __init__(self, params: TrajectoryParams): self._params = params
    @abstractmethod
    def compute_position(self, t: float) -> np.ndarray: pass
    def _compute_phase(self, t: float) -> float:
        return (t * self._params.frequency * 2 * np.pi + self._params.initial_phase)


class Figure8TrajectoryGenerator(TrajectoryGenerator):
    def compute_position(self, t: float) -> np.ndarray:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(phase)
        dy = self._params.scale[1] * np.sin(phase / 2)
        dz = 0.02 * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)


class SquareTrajectoryGenerator(TrajectoryGenerator):
    def compute_position(self, t: float) -> np.ndarray:
        period = 8.0
        phase = (t % period) / period * 4
        size = self._params.scale[0]
        if phase < 1:
            pos = [size, size * (phase), 0]
        elif phase < 2:
            pos = [size * (2 - phase), -size, 0]
        elif phase < 3:
            pos = [-size, -size * (phase - 2), 0]
        else:
            pos = [-size * (4 - phase), size, 0]
        return self._params.center + np.array(pos)


class LissajousTrajectoryGenerator(TrajectoryGenerator):
    def compute_position(self, t: float) -> np.ndarray:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(3 * phase)
        dy = self._params.scale[1] * np.sin(4 * phase + np.pi / 2)
        dz = 0.02 * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz])


class LocalRobotSimulator(gym.Env):
    def __init__(self, model_path=cfg.DEFAULT_MUJOCO_MODEL_PATH,
                 control_freq=cfg.DEFAULT_CONTROL_FREQ,
                 trajectory_type=TrajectoryType.FIGURE_8,
                 randomize_params=False, **kwargs):
        
        super().__init__()
        
        self.n_joints = cfg.N_JOINTS
        self._dt = 1.0 / control_freq
        self._control_freq = control_freq
        self._randomize_params = randomize_params
        self._tick = 0
        
        # PD gains for local simulation
        self.kp_local = cfg.KP_LOCAL
        self.kd_local = cfg.KD_LOCAL

        # Initial Mujoco setup
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        sim_freq = int(1.0 / self.model.opt.timestep)
        self.n_substeps = sim_freq // control_freq

        # IK Solver
        self.ik_solver = IKSolver(self.model, cfg.JOINT_LIMITS_LOWER, cfg.JOINT_LIMITS_UPPER)

        # Trajectory Generator
        self._params = TrajectoryParams()
        self._trajectory_type = trajectory_type
        generators = {
            TrajectoryType.FIGURE_8: Figure8TrajectoryGenerator,
            TrajectoryType.SQUARE: SquareTrajectoryGenerator,
            TrajectoryType.LISSAJOUS_COMPLEX: LissajousTrajectoryGenerator,
        }
        self._generator = generators[trajectory_type](self._params)

        self._q_current = cfg.INITIAL_JOINT_CONFIG.copy()
        self._qd_current = np.zeros(self.n_joints)
        self._last_q_desired = cfg.INITIAL_JOINT_CONFIG.copy()

        self._reset_mujoco_state(cfg.INITIAL_JOINT_CONFIG)

    def _reset_mujoco_state(self, q_init: np.ndarray) -> None:
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.time = 0.0
        self.data.qpos[:self.n_joints] = q_init
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        q_start = cfg.INITIAL_JOINT_CONFIG.copy()
        self.ik_solver.reset_trajectory(q_start=q_start)
        self._trajectory_time = 0.0
        self._tick = 0
        self._reset_mujoco_state(q_start)
        self._q_current = q_start.copy()
        self._qd_current = np.zeros(self.n_joints)
        self._last_q_desired = q_start.copy()
        return self._q_current.astype(np.float32), {}

    def step(self, action=None):
        self._trajectory_time += self._dt
        self._tick += 1
        t = self._trajectory_time

        # 1-second warmup: stay at initial pose
        if t < cfg.WARM_UP_DURATION:
            cartesian_target = self.data.site_xpos[self.model.site('panda_ee_site').id].copy()
            q_desired = cfg.INITIAL_JOINT_CONFIG.copy()
        else:
            movement_time = t - cfg.WARM_UP_DURATION
            cartesian_target = self._generator.compute_position(movement_time)
            q_desired, ik_success, _ = self.ik_solver.solve(cartesian_target, self._q_current)
            if not ik_success or q_desired is None:
                q_desired = self._last_q_desired.copy()

        # Update visual marker
        try:
            mocap_id = self.model.body('target_marker').mocapid[0]
            self.data.mocap_pos[mocap_id] = cartesian_target
        except KeyError:
            pass

        # PD + Inverse Dynamics — FIXED
        qd_desired = (q_desired - self._last_q_desired) / self._dt
        self._last_q_desired = q_desired.copy()

        q_curr = self.data.qpos[:self.n_joints].copy()
        q_error = q_desired - q_curr
        # NO angle wrapping on joint 6 (continuous roll)
        q_error[:5] = (q_error[:5] + np.pi) % (2*np.pi) - np.pi
        q_error[6] = (q_error[6] + np.pi) % (2*np.pi) - np.pi  # joint 7 optional

        acc = self.kp_local * q_error + self.kd_local * (qd_desired - self.data.qvel[:self.n_joints])

        # CORRECT inverse dynamics — no qpos overwrite!
        self.data.qacc[:self.n_joints] = acc
        mujoco.mj_inverse(self.model, self.data)
        self.data.ctrl[:self.n_joints] = self.data.qfrc_inverse[:self.n_joints].copy()

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._q_current = self.data.qpos[:self.n_joints].copy()
        self._qd_current = self.data.qvel[:self.n_joints].copy()

        return self._q_current.astype(np.float32), self._qd_current.astype(np.float32), 0.0, False, False, {}