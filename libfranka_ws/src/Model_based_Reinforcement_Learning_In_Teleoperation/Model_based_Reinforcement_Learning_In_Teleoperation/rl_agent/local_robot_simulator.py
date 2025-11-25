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
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
import mujoco

# Ensure you have the updated Inertia-Weighted IKSolver
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.inverse_kinematics import IKSolver

from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    DEFAULT_MUJOCO_MODEL_PATH,
    N_JOINTS,
    EE_BODY_NAME,
    TCP_OFFSET,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    KP_LOCAL,
    KD_LOCAL,
    DEFAULT_CONTROL_FREQ,
    TRAJECTORY_CENTER,
    TRAJECTORY_SCALE,
    TRAJECTORY_FREQUENCY,
)

# --- TIMING CONSTANTS ---
PHASE_1_HOLD_INITIAL = 5.0   # Seconds to stay at Master Pose
PHASE_2_MOVE_TIME    = 3.0   # Seconds to move from Master to Trajectory Center
PHASE_3_HOLD_CENTER  = 5.0   # Seconds to wait at Center before starting
# Total Warmup = 5 + 3 + 5 = 13.0 seconds

# Z-Axis Oscillation Amplitude
Z_AMPLITUDE = 0.01 

class TrajectoryType(Enum):
    FIGURE_8 = "figure_8"
    SQUARE = "square"
    LISSAJOUS_COMPLEX = "lissajous_complex" 

@dataclass(frozen=True) 
class TrajectoryParams:
    center: NDArray[np.float64] = field(default_factory=lambda: TRAJECTORY_CENTER.copy())
    scale: NDArray[np.float64] = field(default_factory=lambda: TRAJECTORY_SCALE.copy())
    frequency: float = TRAJECTORY_FREQUENCY
    initial_phase: float = 0.0

    def __post_init__(self) -> None:
        assert self.center.shape == (3,)
        assert self.scale.shape == (2,)

class TrajectoryGenerator(ABC):
    def __init__(self, params: TrajectoryParams):
        self._params = params
    @property
    def params(self) -> TrajectoryParams:
        return self._params
    @abstractmethod
    def compute_position(self, t: float) -> NDArray[np.float64]:
        pass
    def _compute_phase(self, t: float) -> float:
        return (t * self._params.frequency * 2 * np.pi + self._params.initial_phase)

class SquareTrajectoryGenerator(TrajectoryGenerator):
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        t_norm = (phase % (2 * np.pi)) / (2 * np.pi)
        corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        segment = int(t_norm * 4) % 4
        segment_progress = (t_norm * 4) % 1
        current_corner = corners[segment]
        next_corner = corners[(segment + 1) % 4]
        smooth_progress = 0.5 * (1 - np.cos(segment_progress * np.pi))
        position_2d = current_corner + smooth_progress * (next_corner - current_corner)
        dx = self._params.scale[0] * position_2d[0]
        dy = self._params.scale[1] * position_2d[1]
        dz = Z_AMPLITUDE * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)

class LissajousComplexGenerator(TrajectoryGenerator):
    _FREQ_RATIO_X = 3.0
    _FREQ_RATIO_Y = 4.0
    _PHASE_SHIFT = np.pi / 4
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(self._FREQ_RATIO_X * phase + self._PHASE_SHIFT)
        dy = self._params.scale[1] * np.sin(self._FREQ_RATIO_Y * phase)
        dz = Z_AMPLITUDE * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)

class Figure8TrajectoryGenerator(TrajectoryGenerator):
    def compute_position(self, t: float) -> NDArray[np.float64]:
        phase = self._compute_phase(t)
        dx = self._params.scale[0] * np.sin(phase)
        dy = self._params.scale[1] * np.sin(phase / 2)
        dz = Z_AMPLITUDE * np.sin(phase)
        return self._params.center + np.array([dx, dy, dz], dtype=np.float64)

class LocalRobotSimulator(gym.Env):
    def __init__(
        self,
        model_path: str = DEFAULT_MUJOCO_MODEL_PATH,
        control_freq: int = DEFAULT_CONTROL_FREQ,
        trajectory_type: TrajectoryType = TrajectoryType.FIGURE_8,
        randomize_params: bool = False,
        joint_limits_lower: Optional[NDArray[np.float64]] = JOINT_LIMITS_LOWER,
        joint_limits_upper: Optional[NDArray[np.float64]] = JOINT_LIMITS_UPPER,
        kp_local: NDArray[np.float64] = KP_LOCAL,
        kd_local: NDArray[np.float64] = KD_LOCAL,
        # warm_up_duration argument is effectively replaced by the constants above, 
        # but kept for signature compatibility if needed.
        warm_up_duration: float = 0.0, 
    ) -> None:
        super().__init__()
        
        self.n_joints = N_JOINTS
        self.ee_body_name = EE_BODY_NAME
        self.tcp_offset = TCP_OFFSET.copy()
        self._dt = 1.0 / control_freq
        self._control_freq = control_freq
        self._randomize_params = randomize_params
        self._tick = 0
        self.kd_local = kd_local.copy()
        self.kp_local = kp_local.copy()
        self.joint_limits_lower = joint_limits_lower.copy()
        self.joint_limits_upper = joint_limits_upper.copy()
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        sim_freq = int(1.0 / self.model.opt.timestep)
        self.n_substeps = sim_freq // control_freq

        # --- MASTER STARTING POSE ---
        # Handshake/Ready Configuration
        self.master_start_pose = np.array([
            0.0,    -0.785, 0.0,    -2.356, 0.0,    1.571,  0.785   
        ], dtype=np.float32)

        # IK Solver
        self.ik_solver = IKSolver(
            model=self.model,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper,
        )

        self.observation_space = gym.spaces.Box(
            low=self.joint_limits_lower.astype(np.float32),
            high=self.joint_limits_upper.astype(np.float32),
            shape=(self.n_joints,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(1)
        self._trajectory_time = 0.0
        self._params = TrajectoryParams()
        self._trajectory_type = trajectory_type
        self._generator = self._create_generator(trajectory_type, self._params)
        
        # State Vectors
        self._q_current = self.master_start_pose.copy()
        self._qd_current = np.zeros(self.n_joints)
        self._last_q_desired = self.master_start_pose.copy()
        
        # To store the Cartesian location of the Master Pose
        self._cartesian_init_pos = np.zeros(3)

    def _create_generator(self, trajectory_type, params):
        generators = {
            TrajectoryType.FIGURE_8: Figure8TrajectoryGenerator,
            TrajectoryType.SQUARE: SquareTrajectoryGenerator,
            TrajectoryType.LISSAJOUS_COMPLEX: LissajousComplexGenerator,
        }
        return generators[trajectory_type](params)

    def _generate_random_params(self):
        return TrajectoryParams(
            center=np.array([np.random.uniform(0.4, 0.5), np.random.uniform(-0.2, 0.2), 0.5]),
            scale=np.array([np.random.uniform(0.1, 0.2), np.random.uniform(0.1, 0.2)]),
            frequency=np.random.uniform(0.05, 0.15),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Reset Solver History
        self.ik_solver.reset_trajectory(q_start=self.master_start_pose)

        if self._randomize_params:
            self._params = self._generate_random_params()
            self._generator = self._create_generator(self._trajectory_type, self._params)

        if options and 'trajectory_type' in options:
            new_type = TrajectoryType(options['trajectory_type'])
            if new_type != self._trajectory_type:
                self._trajectory_type = new_type
                self._generator = self._create_generator(new_type, self._params)

        self._trajectory_time = 0.0
        self._tick = 0
        
        # 2. Determine Cartesian Positions
        # A. Where does the trajectory start?
        self._traj_start_pos = self._generator.compute_position(0.0)
        
        # B. Where is our Master Pose? (Forward Kinematics)
        # We use this to hold position during Phase 1
        self.ik_solver.data.qpos[:self.n_joints] = self.master_start_pose
        mujoco.mj_forward(self.ik_solver.model, self.ik_solver.data)
        # Get site position (requires site name logic similar to get_cartesian_position)
        try:
            site_id = self.model.site('panda_ee_site').id
            self._cartesian_init_pos = self.ik_solver.data.site_xpos[site_id].copy()
        except KeyError:
            raise ValueError("panda_ee_site not found during reset FK")

        # 3. Reset Physics to Master Pose
        q_start = self.master_start_pose.copy()
        
        self._q_current = q_start.copy()
        self._qd_current = np.zeros(self.n_joints)
        self._last_q_desired = q_start.copy()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = q_start
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)

        info = {
            "trajectory_type": self._trajectory_type.value,
            "joint_pos": self._q_current.copy(),
            "center": self._params.center.copy(),
        }
        return self._q_current.astype(np.float32), info

    def _get_inverse_dynamics(self, q, v, a_desired):
        self.data.qpos[:self.n_joints] = q
        self.data.qvel[:self.n_joints] = v
        self.data.qacc[:self.n_joints] = a_desired
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:self.n_joints].copy()
    
    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def step(self, action=None):
        self._trajectory_time += self._dt
        self._tick += 1
        
        # --- 4-STAGE TARGET GENERATION ---
        t = self._trajectory_time
        T1 = PHASE_1_HOLD_INITIAL
        T2 = T1 + PHASE_2_MOVE_TIME
        T3 = T2 + PHASE_3_HOLD_CENTER
        
        if t < T1:
            # PHASE 1: Hold Master Pose
            # We use the Cartesian position of the master pose to keep the IK solver active and stable
            cartesian_target = self._cartesian_init_pos.copy()
            
        elif t < T2:
            # PHASE 2: Move from Master to Trajectory Start (Linear Interpolation)
            progress = (t - T1) / PHASE_2_MOVE_TIME
            # Simple Lerp
            cartesian_target = (1 - progress) * self._cartesian_init_pos + progress * self._traj_start_pos
            
        elif t < T3:
            # PHASE 3: Hold at Trajectory Start Center
            cartesian_target = self._traj_start_pos.copy()
            
        else:
            # PHASE 4: Execute Trajectory
            movement_time = t - T3
            cartesian_target = self._generator.compute_position(movement_time)

        # --- VISUAL DEBUG ---
        try:
            mocap_id = self.model.body('target_marker').mocapid[0]
            self.data.mocap_pos[mocap_id] = cartesian_target
        except KeyError:
            pass 

        # --- SOLVE IK ---
        # Note: We seed with current position to ensure continuity
        q_desired, ik_success, ik_error = self.ik_solver.solve(
            target_pos=cartesian_target,
            q_init=self._q_current, 
            body_name=self.ee_body_name,
            tcp_offset=self.tcp_offset,
            enforce_continuity=True,
        )

        if not ik_success or q_desired is None:
            if self._tick % 100 == 0:
                print(f"IK failed at t={self._trajectory_time:.3f}s, error={ik_error:.6f}m")
            q_desired = self._last_q_desired.copy()

        qd_desired = (q_desired - self._last_q_desired) / self._dt
        self._last_q_desired = q_desired.copy()

        q_current = self.data.qpos[:self.n_joints].copy()
        qd_current = self.data.qvel[:self.n_joints].copy()

        q_error = self._normalize_angle(q_desired - q_current)
        qd_error = qd_desired - qd_current

        acc_desired = self.kp_local * q_error + self.kd_local * qd_error
        tau_control = self._get_inverse_dynamics(q_current, qd_current, acc_desired)
        
        self.data.ctrl[:self.n_joints] = tau_control
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        self._q_current = self.data.qpos[:self.n_joints].copy()
        self._qd_current = self.data.qvel[:self.n_joints].copy()

        cartesian_achieved = self.get_cartesian_position()
        tracking_error = np.linalg.norm(cartesian_target - cartesian_achieved)

        info = {
            "time": self._trajectory_time,
            "trajectory_type": self._trajectory_type.value,
            "cartesian_target": cartesian_target,
            "cartesian_achieved": cartesian_achieved,
            "tracking_error": tracking_error,
            "joint_pos": self._q_current.copy(),
            "joint_vel": self._qd_current.copy(),
            "ik_success": ik_success,
            "ik_error": ik_error,
        }

        return self._q_current.astype(np.float32), self._qd_current.astype(np.float32), 0.0, False, False, info

    def get_joint_state(self):
        return {"joint_pos": self._q_current.copy(), "joint_vel": self._qd_current.copy()}

    def get_cartesian_position(self):
        try:
            site_id = self.model.site('panda_ee_site').id
            return self.data.site_xpos[site_id].copy()
        except KeyError:
            ee_id = self.model.body(self.ee_body_name).id
            flange_pos = self.data.xpos[ee_id].copy()
            flange_rot = self.data.xmat[ee_id].reshape(3, 3)
            return flange_pos + flange_rot @ self.tcp_offset

    def get_position_at_time(self, t):
        # For external queries (like graphs), we might need to adapt this
        # But usually this is used for reference comparison
        # Simulating the same logic as step():
        T1 = PHASE_1_HOLD_INITIAL
        T2 = T1 + PHASE_2_MOVE_TIME
        T3 = T2 + PHASE_3_HOLD_CENTER
        
        if t < T1:
            return self._cartesian_init_pos.copy()
        elif t < T2:
            progress = (t - T1) / PHASE_2_MOVE_TIME
            return (1 - progress) * self._cartesian_init_pos + progress * self._traj_start_pos
        elif t < T3:
            return self._traj_start_pos.copy()
        else:
            return self._generator.compute_position(t - T3)
    
    def get_current_tick(self):
        return self._tick