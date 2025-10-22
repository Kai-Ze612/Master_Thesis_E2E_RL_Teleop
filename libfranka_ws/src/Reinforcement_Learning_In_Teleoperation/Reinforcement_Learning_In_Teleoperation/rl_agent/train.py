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
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback)

# Custom imports
from Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType


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
        randomize_trajectory=args.randomize_trajectory and not for_eval,
        obs_mode=args.obs_mode  # ← ADD THIS LINE
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

    env = make_vec_env(
        lambda: make_env(args, for_eval=False),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv
    )
    eval_env = make_env(args, for_eval=True)
    
    logger.info("Environments created successfully.")

    # Setup Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=model_dir,
        name_prefix="ppo_checkpoint"
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

    # Policy Keyword Arguments
    policy_kwargs = dict(
        net_arch=dict(pi=args.net_arch, vf=args.net_arch), # Use same arch for policy and value
        lstm_hidden_size=args.lstm_size,
        enable_critic_lstm=True # Use LSTM for the critic as well
    )

    model = RecurrentPPO(
        'MlpLstmPolicy',
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs # Pass the new kwargs
    )
    logger.info(f"RecurrentPPO (MlpLstmPolicy) model created with LSTM size {args.lstm_size}.")

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
    exp_group.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments for PPO.")
    exp_group.add_argument(
        "--obs-mode", 
        type=str, 
        default="minimal", 
        choices=["minimal", "context", "full"],
        help="Observation space mode: minimal (51d, recommended), context (93d), full (243d)"
    )
    
    # Policy Configuration
    policy_group = parser.add_argument_group('Policy Configuration')
    policy_group.add_argument("--lstm-size", type=int, default=256, help="Size of the LSTM hidden state.")
    policy_group.add_argument("--net-arch", type=int, nargs='+', default=[512, 256], help="Network architecture (hidden layer sizes) for MLP *after* the LSTM.")

    # --- RecurrentPPO Hyperparameters ---
    ppo_group = parser.add_argument_group('PPO Hyperparameters')
    ppo_group.add_argument("--learning-rate", "--lr", type=float, default=3e-4, help="Learning rate for the optimizer.")
    ppo_group.add_argument("--n-steps", type=int, default=2048, help="Number of steps to run for each environment per update (rollout buffer size).")
    ppo_group.add_argument("--batch-size", type=int, default=64, help="Minibatch size for PPO epochs.")
    ppo_group.add_argument("--n-epochs", type=int, default=10, help="Number of epochs to update the policy per rollout.")
    ppo_group.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    ppo_group.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter for advantage estimation.")
    ppo_group.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping parameter.")
    ppo_group.add_argument("--ent-coef", type=float, default=1e-4, help="Entropy coefficient for exploration.")

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