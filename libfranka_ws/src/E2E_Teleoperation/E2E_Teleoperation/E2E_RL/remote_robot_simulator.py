"""
MuJoCo-based simulator for the remote robot (follower).

Pipelines:
1. Subscribe to predicted local robot state (for error calculation)
2. Subscribe to true local robot state (for error calculation)
3. Subscribe to RL output tau (RL made decision)
4. Step the MuJoCo simulation.
5. Subscribe to remote robot state.
"""

from __future__ import annotations
import heapq
import logging
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import E2E_Teleoperation.config.robot_config as cfg

logger = logging.getLogger(__name__)

class RemoteRobotSimulator:
    def __init__(
        self,
        delay_config: ExperimentConfig = ExperimentConfig.HIGH_VARIANCE,
        seed: Optional[int] = None,
        render: bool = False,
        render_fps: int = 60,
        verbose: bool = True
    ):
        
        self._verbose = verbose
        self._render_enabled = render
        self._render_fps = render_fps
        
        # 1. Load MuJoCo Model (Centralized Path)
        self.model_path = str(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"MuJoCo model not found at: {self.model_path}")

        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}")

        # 2. Initialize Data Structures
        self.data = mujoco.MjData(self.model)
        self._data_control = mujoco.MjData(self.model)

        # 3. Setup Simulation Parameters (Centralized Config)
        self.control_freq = cfg.CONTROL_FREQ
        sim_timestep = self.model.opt.timestep
        self._n_substeps = int(1.0 / (sim_timestep * self.control_freq))

        # 4. Robot Parameters (Centralized Config)
        self.torque_limits = cfg.TORQUE_LIMITS.copy()
        self.n_joints = cfg.N_JOINTS

        # 5. Delay Simulation State
        self.delay_simulator = DelaySimulator(self.control_freq, config=delay_config, seed=seed)
        self._action_queue: List[Tuple[int, np.ndarray]] = []
        self._internal_tick = 0
        self._last_executed_torque = np.zeros(self.n_joints)

        # 6. Warmup Configuration
        total_grace_time = cfg.WARM_UP_DURATION + cfg.NO_DELAY_DURATION
        self._no_delay_steps = int(total_grace_time * self.control_freq)

        # 7. Initialize Viewer
        self._viewer = None
        self._render_interval = max(1, self.control_freq // self._render_fps)
        if self._render_enabled:
            self._init_viewer()

    def _init_viewer(self) -> None:
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.cam.azimuth = 135
        self._viewer.cam.elevation = -20
        self._viewer.cam.distance = 2.0
        self._viewer.cam.lookat[:] = [0.4, 0.0, 0.4]

    def _reset_mujoco_data(self, mj_data: mujoco.MjData, q_init: NDArray[np.float64]) -> None:
        mujoco.mj_resetData(self.model, mj_data)
        mj_data.qpos[:self.n_joints] = q_init
        mujoco.mj_forward(self.model, mj_data)

    def reset(self, initial_qpos: NDArray[np.float64]) -> None:
        self._action_queue = []
        heapq.heapify(self._action_queue)
        self._internal_tick = 0
        self._last_executed_torque = np.zeros(self.n_joints)

        self._reset_mujoco_data(self.data, initial_qpos)
        self._reset_mujoco_data(self._data_control, initial_qpos)
        self._stabilize_robot(initial_qpos)

        if self._render_enabled and self._viewer:
            self._viewer.sync()

    def _stabilize_robot(self, q_pos: NDArray[np.float64], steps: int = 100) -> None:
        """
        Transitions the robot from a mathematical initial state to a stable physical state.
        """
        self._data_control.qpos[:self.n_joints] = q_pos
        self._data_control.qvel[:self.n_joints] = 0.0
        self._data_control.qacc[:self.n_joints] = 0.0
       
        mujoco.mj_inverse(self.model, self._data_control)
        gravity_torque = self._data_control.qfrc_inverse[:self.n_joints].copy()

        for _ in range(steps):
            self.data.ctrl[:self.n_joints] = gravity_torque
            self.data.qvel[:] = 0.0
            mujoco.mj_step(self.model, self.data)

    def _normalize_angle(self, angle: NDArray) -> NDArray:
        """
        Turn angle into [-pi, pi]
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def step(
            self,
            target_q: np.ndarray,
            target_qd: np.ndarray,
            torque_input: np.ndarray,
            true_local_q: Optional[np.ndarray] = None,
            predicted_q: Optional[np.ndarray] = None
        ) -> Dict[str, Any]:
            
            self._internal_tick += 1

            # 1. Simulate Network Delay
            if self._internal_tick < self._no_delay_steps:
                delay_steps = 0
            else:
                delay_steps = int(self.delay_simulator.get_action_delay_steps())

            arrival_time = self._internal_tick + delay_steps
            heapq.heappush(self._action_queue, (arrival_time, torque_input.copy()))

            # 2. Retrieve Pending Actions
            while self._action_queue and self._action_queue[0][0] <= self._internal_tick:
                _, self._last_executed_torque = heapq.heappop(self._action_queue)

            # 3. Apply Torque
            tau_clipped = np.clip(self._last_executed_torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl[:self.n_joints] = tau_clipped

            for _ in range(self._n_substeps):
                mujoco.mj_step(self.model, self.data)

            # 4. Rendering
            if self._render_enabled:
                self.render()

            # 5. Metrics & State
            q_current = self.data.qpos[:self.n_joints].copy()
            
            # Ground Truth is the Local Robot State (Leader)
            ground_truth_q = true_local_q if true_local_q is not None else target_q
            
            # Prediction processing (handle if prediction includes velocity)
            pred_q_eval = None
            pred_error = 0.0
            
            if predicted_q is not None:
                # If prediction includes velocities (14 dims), take only positions (7 dims)
                if predicted_q.shape[0] > self.n_joints:
                    pred_q_eval = predicted_q[:self.n_joints]
                else:
                    pred_q_eval = predicted_q
                
                pred_error = np.linalg.norm(self._normalize_angle(ground_truth_q - pred_q_eval))

            # Tracking Error (Remote vs. Local)
            tracking_error = np.linalg.norm(self._normalize_angle(ground_truth_q - q_current))

            # 6. Logging [UPDATED to match requested 4 items]
            if self._verbose:
                self._log_step_info(ground_truth_q, q_current, pred_q_eval, self._last_executed_torque)

            return {
                "joint_error": np.linalg.norm(ground_truth_q - q_current),
                "tracking_error": tracking_error,
                "prediction_error": pred_error,
                "tau_total": self._last_executed_torque
            }

    def _log_step_info(self, local_true_q, remote_q, predicted_q, rl_tau):
        """
        Prints the 4 key metrics requested:
        1. True q (Local)
        2. Predicted q
        3. RL Torque
        4. Remote q
        """
        np.set_printoptions(precision=3, suppress=True, linewidth=120)
        
        print(f"\n[Step {self._internal_tick}] Simulation Status")
        print("=" * 60)
        print(f"1. True q (Local):   {local_true_q}")
        print(f"2. Predicted q:      {predicted_q if predicted_q is not None else 'N/A (Warmup)'}")
        print(f"3. RL Output Torque: {rl_tau}")
        print(f"4. Remote q:         {remote_q}")
        print("=" * 60)

    def render(self) -> bool:
        if not self._render_enabled or self._viewer is None:
            return True
        if not self._viewer.is_running():
            return False
        
        if self._internal_tick % self._render_interval == 0:
            self._viewer.sync()
        return True

    def get_joint_state(self) -> Tuple[NDArray, NDArray]:
        return (
            self.data.qpos[:self.n_joints].copy(),
            self.data.qvel[:self.n_joints].copy()
        )

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()