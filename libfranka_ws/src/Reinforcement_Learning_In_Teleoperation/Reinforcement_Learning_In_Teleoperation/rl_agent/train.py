"""
Training script for the joint-space teleoperation environment.
"""

# Python imports
import os
import sys
from datetime import datetime
import logging
import argparse
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
from Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
from Reinforcement_Learning_In_Teleoperation.rl_agent.custom_policy import get_policy_kwargs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EarlyStoppingCallback(BaseCallback):
    """Early stops training"""
    def __init__(self, eval_freq: int, patience: int, min_improvement: float, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:

            # The EvalCallback should have just run and logged the 'eval/mean_reward'
            current_reward = self.logger.name_to_value.get('eval/mean_reward')

            if current_reward is None:
                return True

            improvement = current_reward - self.best_mean_reward

            if improvement >= self.min_improvement:
                if self.verbose > 0:
                    logger.info(f"EarlyStopping: New best model! Reward improved from {self.best_mean_reward:.2f} to {current_reward:.2f}")
                self.best_mean_reward = current_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                if self.verbose > 0:
                    logger.info(f"EarlyStopping: No significant improvement. Patience: {self.no_improvement_count}/{self.patience}")

            if self.no_improvement_count >= self.patience:
                if self.verbose > 0:
                    logger.warning(f"EARLY STOPPING: No improvement for {self.patience} evaluations. Stopping training.")
                return False  # Returning False stops the training

        return True
    
# Custom Callback for Tracking Metrics
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

def make_env(args: argparse.Namespace, for_eval: bool = False) -> Monitor:
    """Creates and wraps the appropriate environment."""
    env = TeleoperationEnvWithDelay(
        experiment_config=args.config,
        trajectory_type=args.trajectory_type,
        randomize_trajectory=args.randomize_trajectory and not for_eval
    )
    return Monitor(env)

def train_agent(args: argparse.Namespace) -> None:
    """Sets up and runs the SAC training loop based on parsed arguments."""
 
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = args.config.name
    trajectory_name = args.trajectory_type.value
    run_name = f"{config_name}_{trajectory_name}_{timestamp}"
    output_dir = os.path.join(args.output_path, run_name)
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training on: {args.trajectory_type.value} with {args.config.name}")

    logger.info("Creating training and evaluation environments...")
    env = make_env(args, for_eval=False)
    eval_env = make_env(args, for_eval=True)
    logger.info("Environments created successfully.")

    # Setup Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=model_dir,
        name_prefix="sac_checkpoint"
    )
    callbacks = [eval_callback, checkpoint_callback, TrackingMetricsCallback()]

    # Early Stopping Callback
    if args.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            eval_freq=args.eval_freq,
            patience=args.patience,
            min_improvement=args.min_improvement
        )
        callbacks.append(early_stopping_callback)
        logger.info(f"Early stopping enabled: patience={args.patience}, min_improvement={args.min_improvement}")
    
    policy_kwargs = get_policy_kwargs(args.policy_type, args.net_arch)

    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ent_coef='auto',
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=args.learning_starts,
        seed=args.seed,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs
    )
    logger.info("SAC model created.")

    # Train models
    try:
        logger.info("Starting training...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="SAC_JointSpace"
        )
        model.save(os.path.join(model_dir, "final_model.zip"))
        logger.info("Training completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        model.save(os.path.join(model_dir, "interrupted_model.zip"))
    finally:
        env.close()
        eval_env.close()
        logger.info("Environments closed.")

def main():
    """Parses command-line arguments and launches the training process."""
    parser = argparse.ArgumentParser(
        description="Train an SAC agent for joint-space teleoperation with delay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment Configuration
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument("--config", type=int, default=2, choices=[1, 2, 3, 4], help="Delay config: 1=LOW, 2=MEDIUM, 3=HIGH, 4=NO_DELAY")
    exp_group.add_argument("--trajectory-type", type=str, default="figure_8", choices=["figure_8", "square", "lissajous_complex"],help="Type of reference trajectory.")
    exp_group.add_argument("--randomize-trajectory", action="store_true", help="If set, randomize trajectory parameters for better generalization.")
    exp_group.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps.")
    exp_group.add_argument("--output-path", type=str, default="./rl_training_output", help="Directory to save models and logs.")
    exp_group.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    # Policy Configuration
    policy_group = parser.add_argument_group('Policy Configuration')
    policy_group.add_argument("--policy-type", type=str, default="baseline", choices=["baseline", "interpolation", "learned_predictor"])

    # --- SAC Hyperparameters ---
    sac_group = parser.add_argument_group('SAC Hyperparameters')
    sac_group.add_argument("--learning-rate", "--lr", type=float, default=3e-4, help="Learning rate for the optimizer.")
    sac_group.add_argument("--buffer-size", type=int, default=200_000, help="Size of the replay buffer.")
    sac_group.add_argument("--batch-size", type=int, default=512, help="Minibatch size for each training step.")
    sac_group.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    sac_group.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient for target networks.")
    sac_group.add_argument("--learning-starts", type=int, default=10_000, help="Number of steps to collect before training starts.")
    sac_group.add_argument("--net-arch", type=int, nargs='+', default=[512, 256], help="Network architecture (hidden layer sizes).")

    # --- Callback Configuration ---
    cb_group = parser.add_argument_group('Callback Configuration')
    cb_group.add_argument("--eval-freq", type=int, default=5000, help="Frequency of evaluation.")
    cb_group.add_argument("--save-freq", type=int, default=50000, help="Frequency of saving model checkpoints.")
    cb_group.add_argument("--early-stopping", action="store_false", help="Enable early stopping if performance stagnates.")
    cb_group.add_argument("--patience", type=int, default=10, help="Patience for early stopping (number of evaluations).")
    cb_group.add_argument("--min-improvement", type=float, default=30.0, help="Minimum reward improvement to reset patience.")
    
    args = parser.parse_args()

    # --- Process Arguments ---
    config_map = {
        1: ExperimentConfig.LOW_DELAY,
        2: ExperimentConfig.MEDIUM_DELAY,
        3: ExperimentConfig.HIGH_DELAY,
        4: ExperimentConfig.NO_DELAY_BASELINE,
    }
    args.config = config_map[args.config]

    trajectory_map = {
        "figure_8": TrajectoryType.FIGURE_8,
        "square": TrajectoryType.SQUARE,
        "lissajous_complex": TrajectoryType.LISSAJOUS_COMPLEX,
    }
    args.trajectory_type = trajectory_map[args.trajectory_type]

    # --- Launch Training ---
    train_agent(args)


if __name__ == "__main__":
    main()