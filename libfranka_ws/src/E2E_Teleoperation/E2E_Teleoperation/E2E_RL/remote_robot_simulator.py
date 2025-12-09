"""
MuJoCo-based simulator for the remote robot (follower).

Pipelines:
1. Subscribe to predicted local robot state (for error calculation).
2. Subscribe to direct torque command from RL. (for moving to the desired state) (the RL output is the goal tau)
3. Step the MuJoCo simulation.
"""


from __future__ import annotations
from typing import Tuple, Optional, List
import heapq
import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from E2E_Teleoperation.E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import E2E_Teleoperation.E2E_Teleoperation.config.robot_config as cfg


class RemoteRobotSimulator:
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.HIGH_VARIANCE,
        seed: Optional[int] = None,
        render: bool = False,
        render_fps: Optional[int] = 120
    ):
        
        # Load MuJoCo model
        self.model_path = cfg.DEFAULT_MUJOCO_MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        
        # Separate Data: One for Physics, One for Math
        self.data = mujoco.MjData(self.model)
        self.data_control = mujoco.MjData(self.model)
        
        # Control Parameters
        self.control_freq = cfg.DEFAULT_CONTROL_FREQ
        sim_freq = int(1.0 / self.model.opt.timestep)
        self.n_substeps = sim_freq // self.control_freq

        # Robot Parameters
        self.torque_limits = cfg.TORQUE_LIMITS.copy()
        self.n_joints = cfg.N_JOINTS
        
        # Simulation State
        self.delay_simulator = DelaySimulator(self.control_freq, config=delay_config, seed=seed)
        self.action_queue: List[Tuple[int, np.ndarray]] = []  # For storing actions in delayed sequence
        self.internal_tick = 0
        self.last_executed_rl_torque = np.zeros(self.n_joints)

        # Warmup (No delay initially)
        total_grace_time = cfg.WARM_UP_DURATION + cfg.NO_DELAY_DURATION
        self.no_delay_steps = int(total_grace_time * self.control_freq)
        
        # Rendering
        self._render_enabled = render
        self._viewer = None
        self._render_fps = render_fps
        self._render_interval = max(1, self.control_freq // self._render_fps)
        
        if self._render_enabled:
            self._init_viewer()

    def _init_viewer(self) -> None:
        """Initialize the MuJoCo passive viewer."""
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.cam.azimuth = 135
        self._viewer.cam.elevation = -20
        self._viewer.cam.distance = 2.0
        self._viewer.cam.lookat[:] = [0.4, 0.0, 0.4]
        print("MuJoCo Viewer initialized.")

    def _reset_mujoco_data(self, mj_data: mujoco.MjData, q_init: NDArray[np.float64]) -> None:
        """Reset a MuJoCo data structure to a specific joint configuration."""
        mj_data.qvel[:] = 0.0
        mj_data.qacc[:] = 0.0
        mj_data.qacc_warmstart[:] = 0.0
        mj_data.ctrl[:] = 0.0
        mj_data.qfrc_applied[:] = 0.0
        mj_data.xfrc_applied[:] = 0.0
        mj_data.time = 0.0
        mj_data.qpos[:self.n_joints] = q_init
        mujoco.mj_forward(self.model, mj_data)
        
    def reset(self, initial_qpos: NDArray[np.float64]) -> None:
        
        # Reset Internal State
        self.action_queue = []
        heapq.heapify(self.action_queue)
        self.internal_tick = 0
        self.last_executed_rl_torque = np.zeros(self.n_joints)
        
        # Reset Physics Data
        self._reset_mujoco_data(self.data, initial_qpos)
        self._reset_mujoco_data(self.data_control, initial_qpos)

        # Calculate Gravity Compensation (Just for settling)
        # We still need data_control here to calculate the torque needed to hold still
        self.data_control.qpos[:self.n_joints] = initial_qpos
        self.data_control.qvel[:self.n_joints] = 0.0
        self.data_control.qacc[:self.n_joints] = 0.0
        mujoco.mj_inverse(self.model, self.data_control)
        gravity_torque = self.data_control.qfrc_inverse[:self.n_joints].copy()
        
        # Settling Loop (Anti-Shock)
        for _ in range(100):
            self.data.ctrl[:self.n_joints] = gravity_torque
            self.data.qvel[:self.n_joints] = 0.0 
            mujoco.mj_step(self.model, self.data)
        
        if self._render_enabled and self._viewer is not None:
            self._viewer.sync()

    def _normalize_angle(self, angle: np.ndarray) -> np.ndarray:
        """Because joint angles can wrap around, normalize to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def step(
        self,
        target_q: np.ndarray,
        target_qd: np.ndarray,
        torque_input: np.ndarray, 
        true_local_q: Optional[np.ndarray] = None
    ) -> dict:
        
        self.internal_tick += 1
        
        # 1. Handle Delay
        if self.internal_tick < self.no_delay_steps:
            delay_steps = 0
        else:
            delay_steps = int(self.delay_simulator.get_action_delay_steps())
            
        arrival_time = self.internal_tick + delay_steps
        
        # Queue the RL action
        heapq.heappush(self.action_queue, (arrival_time, torque_input.copy()))
        
        # Retrieve arrived action
        while self.action_queue and self.action_queue[0][0] <= self.internal_tick:
            _, self.last_executed_rl_torque = heapq.heappop(self.action_queue)
            
        # Apply tau
        tau_total = self.last_executed_rl_torque
        tau_clipped = np.clip(tau_total, -self.torque_limits, self.torque_limits)
        
        self.data.ctrl[:self.n_joints] = tau_clipped
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Rendering
        if self._render_enabled:
            self.render()

        # Metrics
        if true_local_q is not None:
            raw_pred_diff = target_q - true_local_q
            prediction_error_norm = np.linalg.norm(self._normalize_angle(raw_pred_diff))
        else:
            prediction_error_norm = 0.0

        q_current = self.data.qpos[:self.n_joints].copy()
        raw_tracking_diff = (true_local_q if true_local_q is not None else target_q) - q_current
        tracking_error_norm = np.linalg.norm(self._normalize_angle(raw_tracking_diff))
        
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print(f"\n[Step {self.internal_tick}]")
        print(f"  Target/Pred Q: {target_q}")
        print(f"  Remote Q:      {q_current}")
        print(f"  RL Torque:     {tau_total}")
        print(f"  Track Error:   {tracking_error_norm:.4f}")
        
        return {
            "joint_error": np.linalg.norm(target_q - self.data.qpos[:self.n_joints]),
            "tracking_error": tracking_error_norm,
            "prediction_error": prediction_error_norm,
            "tau_pd": np.zeros(self.n_joints), 
            "tau_rl": self.last_executed_rl_torque, 
            "tau_total": tau_total
        }
        
    def render(self) -> bool:
        if not self._render_enabled or self._viewer is None: return True
        if not self._viewer.is_running(): return False
        if self.internal_tick % self._render_interval == 0: self._viewer.sync()
        return True

    def get_joint_state(self):
        return self.data.qpos[:self.n_joints].copy(), self.data.qvel[:self.n_joints].copy()
    
    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
    
    def __del__(self):
        self.close()