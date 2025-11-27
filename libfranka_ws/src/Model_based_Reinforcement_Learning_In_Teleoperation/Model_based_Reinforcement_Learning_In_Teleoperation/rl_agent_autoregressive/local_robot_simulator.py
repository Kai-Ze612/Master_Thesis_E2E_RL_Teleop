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
    center: np.ndarray = field(default_factory=lambda: cfg.TRAJECTORY_CENTER.copy())
    scale: np.ndarray = field(default_factory=lambda: cfg.TRAJECTORY_SCALE.copy())
    frequency: float = cfg.TRAJECTORY_FREQUENCY
    initial_phase: float = 0.0

    @classmethod
    def randomized(cls, actual_start_pos: np.ndarray) -> TrajectoryParams:
        center_x = np.random.uniform(0.3, 0.4)
        center_y = np.random.uniform(-0.1, 0.1)
        center_z = actual_start_pos[2]
        
        center = np.array([center_x, center_y, center_z], dtype=np.float64)
        
        scale_x = np.random.uniform(0.1, 0.1)
        scale_y = np.random.uniform(0.1, 0.3)
        scale_z = 0.02
        
        scale = np.array([scale_x, scale_y, scale_z], dtype=np.float64)
        
        frequency = np.random.uniform(0.05, 0.15)
        return cls(center=center, scale=scale, frequency=frequency, initial_phase=0.0)


class TrajectoryGenerator(ABC):
    def __init__(self, params: TrajectoryParams):
        self._params = params
    
    @abstractmethod
    def compute_position(self, t: float) -> np.ndarray:
        pass
    
    def _compute_phase(self, t: float) -> float:
        return t * self._params.frequency * 2 * np.pi + self._params.initial_phase


class Figure8TrajectoryGenerator(TrajectoryGenerator):
    def compute_position(self, t: float) -> np.ndarray:

        phase = self._compute_phase(t)

        dx = self._params.scale[0] * np.sin(phase)

        dy = self._params.scale[1] * np.sin(phase / 2)

        dz = self._params.scale[2] * np.sin(phase)

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
        
        # Load model for IK/FK only
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Setup IK Solver
        self.ik_solver = IKSolver(self.model, cfg.JOINT_LIMITS_LOWER, cfg.JOINT_LIMITS_UPPER)
        
        # Get start position via FK
        self.data.qpos[:self.n_joints] = cfg.INITIAL_JOINT_CONFIG
        mujoco.mj_forward(self.model, self.data)
        ee_site_id = self.model.site('panda_ee_site').id
        self.actual_spawn_pos = self.data.site_xpos[ee_site_id].copy()
        
        # Trajectory Generator Setup
        if self._randomize_params:
            self._params = TrajectoryParams.randomize(self.actual_spawn_pos)
        else:
            self._params = TrajectoryParams()
        self._trajectory_type = trajectory_type
       
        # Trajectory Generator Selection
        generators = {
            TrajectoryType.FIGURE_8: Figure8TrajectoryGenerator,
            TrajectoryType.SQUARE: SquareTrajectoryGenerator,
            TrajectoryType.LISSAJOUS_COMPLEX: LissajousTrajectoryGenerator,
        }
        self._generator = generators[trajectory_type](self._params)
        self.traj_start_pos = self._generator.compute_position(0.0)
        
        # State
        self._q_start = cfg.INITIAL_JOINT_CONFIG.copy()
        self._q_current = self._q_start.copy()
        self._q_previous = self._q_start.copy()
        self._trajectory_time = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._trajectory_time = 0.0
        self._tick = 0
        self._q_current = self._q_start.copy()
        self._q_previous = self._q_start.copy()
        
        self.ik_solver.reset_trajectory(q_start=self._q_start)
        
        return self._q_current.astype(np.float32), {}

    def step(self, action=None):
        """
        Generate next desired joint positions and velocities
        
        Returns:
        - q_desired: Desired joint positions
        - qd_desired: Desired joint velocities
        - reward: Always 0.0 (not used)
        - terminated: Always False
        - truncated: Always False
        - info: Empty dict
        """
        
        self._trajectory_time += self._dt
        self._tick += 1
        t = self._trajectory_time

        # 1. Get the Raw Target from Trajectory/IK (Your existing logic)
        if t < cfg.WARM_UP_DURATION:
            progress = t / cfg.WARM_UP_DURATION
            current_target_pos = (1 - progress) * self.actual_spawn_pos + progress * self.traj_start_pos
            q_target_raw, ik_success, _ = self.ik_solver.solve(current_target_pos, self._q_current)
        else:
            movement_time = t - cfg.WARM_UP_DURATION
            cartesian_target = self._generator.compute_position(movement_time)
            q_target_raw, ik_success, _ = self.ik_solver.solve(cartesian_target, self._q_current)
            
        if not ik_success or q_target_raw is None:
            q_target_raw = self._q_current.copy()
        
        qd_raw = (q_target_raw - self._q_previous) / self._dt
        
        max_velocity = 1.5 
        qd_safe = np.clip(qd_raw, -max_velocity, max_velocity)
  
        q_safe = self._q_previous + (qd_safe * self._dt)
        
        # Update state
        self._q_previous = self._q_current.copy()
        self._q_current = q_safe.copy() # Store the SAFE position, not the RAW one
        
        return (
            self._q_current.astype(np.float32),   
            qd_safe.astype(np.float32),        
            0.0, 
            False,
            False,
            {}
        )
    
    @property
    def trajectory_time(self):
        return self._trajectory_time
    
    @property
    def current_tick(self):
        """Current tick count."""
        return self._tick