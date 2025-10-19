"""
Training script for the joint-space teleoperation environment.
"""

# Python imports
import os
from datetime import datetime
import logging
import numpy as np

# Stable Baselines3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback)

# Custom imports
from training_env import TeleoperationEnvWithDelay
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from local_robot_simulator import TrajectoryType
from Reinforcement_Learning_In_Teleoperation.config.robot_config import DEFAULT_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressLoggingCallback(BaseCallback):
    """Logs training progress and episode statistics."""
    def __init__(self, log_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # You can add any custom logging here if needed
            pass
        return True
    
class TrackingMetricsCallback(BaseCallback):
    """Logs the real-time joint space tracking error to TensorBoard."""
    def __init__(self, log_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.joint_errors = []

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'real_time_joint_error' in info:
                    self.joint_errors.append(info['real_time_joint_error'])

        # Log the mean of the last N errors to TensorBoard
        if self.n_calls % self.log_freq == 0 and self.joint_errors:
            mean_error = np.mean(self.joint_errors)
            self.logger.record('metrics/real_time_joint_error_mean', mean_error)
            self.joint_errors = [] # Reset buffer

        return True

# Environment Factory Functions
def make_training_env(
    config: ExperimentConfig,
    trajectory: TrajectoryType,
    randomize: bool
) -> Monitor:
    """Creates and wraps the training environment."""
    env = TeleoperationEnvWithDelay(
        experiment_config=config,
        trajectory_type=trajectory,
        randomize_trajectory=randomize
    )
    return Monitor(env)

def make_evaluation_env(
    config: ExperimentConfig,
    trajectory: TrajectoryType
) -> Monitor:
    """Creates and wraps the evaluation environment."""
    env = TeleoperationEnvWithDelay(
        experiment_config=config,
        trajectory_type=trajectory,
        randomize_trajectory=False  # Never randomize for evaluation
    )
    return Monitor(env)

# Training Function
def train_agent(
    experiment_config: ExperimentConfig,
    trajectory_type: TrajectoryType,
    randomize_trajectory: bool,
    total_timesteps: int,
    output_path: str,
    seed: Optional[int]
) -> None: