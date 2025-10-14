"""
Training script for teleoperation with adaptive control and learned NN predictor.

This script trains an SAC agent with:
- Adaptive PD controller (delay-aware gain scheduling)
- IK solver with trajectory continuity
- Neural network predictor (integrated in custom policy)
"""

# Python imports
import os
import sys
import argparse
from datetime import datetime
import torch.nn as nn
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

from custom_policy import (
    create_predictor_policy,
    create_interpolation_policy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Custom Callbacks
# ============================================================

class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback based on evaluation performance.

    Stops training if no improvement is observed for a specified number
    of consecutive evaluations.
    """

    def __init__(
        self,
        eval_freq: int,
        patience: int = 10,
        min_improvement: float = 1.0,
        verbose: int = 1
    ):
        """
        Args:
            eval_freq: Frequency of evaluation (in steps)
            patience: Number of evaluations without improvement before stopping
            min_improvement: Minimum reward improvement to reset patience counter
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.eval_count = 0

    def _on_step(self) -> bool:
        """Called after each environment step."""
        if self.n_calls % self.eval_freq != 0:
            return True

        self.eval_count += 1
        current_reward = self.logger.name_to_value.get('eval/mean_reward', None)

        if current_reward is None:
            return True

        improvement = current_reward - self.best_mean_reward

        if improvement > self.min_improvement:
            logger.info(f"\n{'='*70}")
            logger.info(f"NEW BEST MODEL")
            logger.info(f"  Previous best: {self.best_mean_reward:.1f}")
            logger.info(f"  Current reward: {current_reward:.1f}")
            logger.info(f"  Improvement: +{improvement:.1f}")
            logger.info(f"{'='*70}\n")

            self.best_mean_reward = current_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            logger.info(
                f"No improvement: {self.no_improvement_count}/{self.patience} "
                f"(current: {current_reward:.1f}, best: {self.best_mean_reward:.1f})"
            )

            if self.no_improvement_count >= self.patience:
                logger.info(f"\n{'='*70}")
                logger.info(f"EARLY STOPPING TRIGGERED")
                logger.info(f"  No improvement for {self.patience} evaluations")
                logger.info(f"  Best reward: {self.best_mean_reward:.1f}")
                logger.info(f"  Total steps: {self.num_timesteps:,}")
                logger.info(f"{'='*70}\n")
                return False

        return True


class TrackingMetricsCallback(BaseCallback):
    """
    Log detailed tracking metrics to TensorBoard.

    Tracks:
    - Cartesian tracking error
    - Delay magnitude and gain adaptation
    - Controller performance (baseline vs. RL correction)
    - IK success rate
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.tracking_errors = []
        self.delays = []
        self.gain_ratios = []
        self.correction_percentages = []
        self.ik_failures = []

    def _on_step(self) -> bool:
        """Called after each environment step."""
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]

            # Tracking error
            if 'real_time_cartesian_error_mm' in info:
                self.tracking_errors.append(info['real_time_cartesian_error_mm'])

            # Delay and gain adaptation
            if 'current_delay_ms' in info:
                self.delays.append(info['current_delay_ms'])
            if 'gain_ratio' in info:
                self.gain_ratios.append(info['gain_ratio'])

            # Controller performance
            if 'mean_correction_percentage' in info:
                self.correction_percentages.append(info['mean_correction_percentage'])

            # IK performance
            if 'ik_success' in info:
                self.ik_failures.append(0 if info['ik_success'] else 1)

            # Log every 100 steps
            if len(self.tracking_errors) % 100 == 0 and len(self.tracking_errors) > 0:
                # Tracking performance
                self.logger.record(
                    'metrics/tracking_error_mm_mean',
                    np.mean(self.tracking_errors[-100:])
                )
                self.logger.record(
                    'metrics/tracking_error_mm_std',
                    np.std(self.tracking_errors[-100:])
                )

                # Delay and adaptation
                if self.delays:
                    self.logger.record(
                        'metrics/delay_ms_mean',
                        np.mean(self.delays[-100:])
                    )
                if self.gain_ratios:
                    self.logger.record(
                        'metrics/gain_ratio_mean',
                        np.mean(self.gain_ratios[-100:])
                    )

                # Controller performance
                if self.correction_percentages:
                    self.logger.record(
                        'metrics/correction_pct_mean',
                        np.mean(self.correction_percentages[-100:])
                    )

                # IK performance
                if self.ik_failures:
                    self.logger.record(
                        'metrics/ik_failure_rate',
                        np.mean(self.ik_failures[-100:])
                    )

        return True


class ProgressLoggingCallback(BaseCallback):
    """
    Log training progress at regular intervals.
    """

    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Log episode statistics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])

        # Periodic progress logging
        if self.n_calls % self.log_freq == 0:
            logger.info(f"\n{'='*70}")
            logger.info(f"TRAINING PROGRESS")
            logger.info(f"  Steps: {self.num_timesteps:,}")

            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-100:]
                logger.info(f"  Recent episodes: {len(recent_rewards)}")
                logger.info(f"  Mean reward: {np.mean(recent_rewards):.2f}")
                logger.info(f"  Std reward: {np.std(recent_rewards):.2f}")
                logger.info(f"  Mean length: {np.mean(self.episode_lengths[-100:]):.0f}")

            logger.info(f"{'='*70}\n")

        return True


# ============================================================
# Environment Factory Functions
# ============================================================

def make_training_env(args: argparse.Namespace):
    """
    Create training environment with specified configuration.

    Args:
        args: Command-line arguments

    Returns:
        Wrapped environment ready for training
    """
    env = TeleoperationEnvWithDelay(
        model_path=args.model_path,
        experiment_config=args.config,
        max_episode_steps=args.max_steps,
        control_freq=args.freq,
        max_cartesian_error=args.max_cartesian_error,
        # Trajectory parameters
        trajectory_type=args.trajectory_type,
        randomize_trajectory=args.randomize_trajectory,
        # Controller parameters
        min_gain_ratio=args.min_gain_ratio,
        delay_threshold=args.delay_threshold,
        # IK parameters
        max_joint_change=args.max_joint_change,
        continuity_gain=args.continuity_gain,
        # History parameters
        target_history_len=args.target_history_len,
        action_history_len=args.action_history_len,
    )

    return Monitor(env)


def make_evaluation_env(args: argparse.Namespace):
    """
    Create evaluation environment (no trajectory randomization).

    Args:
        args: Command-line arguments

    Returns:
        Wrapped environment ready for evaluation
    """
    env = TeleoperationEnvWithDelay(
        model_path=args.model_path,
        experiment_config=args.config,
        max_episode_steps=args.max_steps * 2,  # Longer episodes for evaluation
        control_freq=args.freq,
        max_cartesian_error=args.max_cartesian_error,
        # Trajectory parameters
        trajectory_type=args.trajectory_type,
        randomize_trajectory=False,  # Fixed for reproducibility
        # Controller parameters
        min_gain_ratio=args.min_gain_ratio,
        delay_threshold=args.delay_threshold,
        # IK parameters
        max_joint_change=args.max_joint_change,
        continuity_gain=args.continuity_gain,
        # History parameters
        target_history_len=args.target_history_len,
        action_history_len=args.action_history_len,
    )

    return Monitor(env)


# ============================================================
# Training Function
# ============================================================

def train_agent(args: argparse.Namespace) -> str:
    """
    Train SAC agent with adaptive controllers.

    Args:
        args: Command-line arguments

    Returns:
        output_dir: Path to training output directory
    """

    # ============================================================
    # Setup Directories
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = args.config.name if isinstance(args.config, ExperimentConfig) else f"Config{args.config}"
    trajectory_name = args.trajectory_type.value if isinstance(args.trajectory_type, TrajectoryType) else args.trajectory_type

    run_name = f"{config_name}_{trajectory_name}_AdaptivePD_{timestamp}"
    output_dir = os.path.join(args.output_path, run_name)
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ============================================================
    # Print Configuration
    # ============================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"TELEOPERATION RL TRAINING")
    logger.info(f"{'='*70}")
    logger.info(f"Configuration:")
    logger.info(f"  Delay config: {config_name}")
    logger.info(f"  Trajectory: {trajectory_name}")
    logger.info(f"  Randomize trajectory: {args.randomize_trajectory}")
    logger.info(f"  Total timesteps: {args.timesteps:,}")
    logger.info(f"  Control frequency: {args.freq} Hz")
    logger.info(f"")
    logger.info(f"Controller Configuration:")
    logger.info(f"  Min gain ratio: {args.min_gain_ratio}")
    logger.info(f"  Delay threshold: {args.delay_threshold}s")
    logger.info(f"  Max joint change: {args.max_joint_change} rad")
    logger.info(f"  Continuity gain: {args.continuity_gain}")
    logger.info(f"")
    logger.info(f"RL Algorithm (SAC):")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Buffer size: {args.buffer_size:,}")
    logger.info(f"  Gamma: {args.gamma}")
    logger.info(f"  Tau: {args.tau}")
    logger.info(f"")
    logger.info(f"Output:")
    logger.info(f"  Directory: {output_dir}")
    logger.info(f"  Seed: {args.seed if args.seed else 'Random'}")
    logger.info(f"{'='*70}\n")

    # ============================================================
    # Create Environments
    # ============================================================
    logger.info("Creating training and evaluation environments...")

    if args.n_envs > 1:
        # Vectorized environments for parallel training
        env = make_vec_env(
            lambda: make_training_env(args),
            n_envs=args.n_envs,
            seed=args.seed
        )
    else:
        # Single environment
        env = make_training_env(args)
        if args.seed is not None:
            env.reset(seed=args.seed)

    eval_env = make_evaluation_env(args)

    logger.info(f"Environments created successfully")
    logger.info(f"  Training envs: {args.n_envs}")
    logger.info(f"  Observation space: {env.observation_space.shape}")
    logger.info(f"  Action space: {env.action_space.shape}\n")

    # ============================================================
    # Setup Callbacks
    # ============================================================
    logger.info("Setting up callbacks...")

    callbacks = [
        # Evaluation callback
        EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1
        ),

        # Checkpoint callback
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=model_dir,
            name_prefix="sac_checkpoint"
        ),

        # Tracking metrics callback
        TrackingMetricsCallback(verbose=1),

        # Progress logging callback
        ProgressLoggingCallback(log_freq=10000, verbose=1),
    ]

    # Add early stopping if requested
    if args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                eval_freq=args.eval_freq,
                patience=args.patience,
                min_improvement=args.min_improvement,
                verbose=1
            )
        )
        logger.info(f"  Early stopping: Enabled (patience={args.patience})")

    logger.info(f"Callbacks configured\n")

    # ============================================================
    # Create SAC Model
    # ============================================================
    logger.info("Creating SAC model...")

    # Select policy based on arguments
    if args.policy_type == 'learned_predictor':
        logger.info("  Using LEARNED PREDICTOR policy")
        policy_kwargs = create_predictor_policy(
            features_dim=args.features_dim,
            predictor_arch=args.predictor_arch,
            controller_arch=args.controller_arch,
            actor_arch=args.net_arch,
            critic_arch=args.net_arch,
        )
    elif args.policy_type == 'linear_interpolation':
        logger.info("  Using LINEAR INTERPOLATION baseline policy")
        policy_kwargs = create_interpolation_policy(
            features_dim=args.features_dim
        )
    else:  # 'default'
        logger.info("  Using DEFAULT MLP policy (no prediction)")
        policy_kwargs = {
            "net_arch": args.net_arch,
            "activation_fn": args.activation_fn,
        }

    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=args.learning_starts,
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        seed=args.seed,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs
    )

    logger.info(f"SAC model created")
    logger.info(f"  Policy type: {args.policy_type}")
    logger.info(f"  Policy: {type(model.policy).__name__}")
    logger.info(f"  Network architecture: {args.net_arch}\n")


    # ============================================================
    # Train Model
    # ============================================================
    try:
        logger.info(f"{'='*70}")
        logger.info(f"STARTING TRAINING")
        logger.info(f"{'='*70}\n")

        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="SAC_AdaptivePD"
        )

        # Save final model
        final_model_path = os.path.join(model_dir, "final_model.zip")
        model.save(final_model_path)

        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*70}")
        logger.info(f"Final model: {final_model_path}")
        logger.info(f"Best model: {os.path.join(model_dir, 'best_model.zip')}")
        logger.info(f"Logs: {log_dir}")
        logger.info(f"{'='*70}\n")

    except KeyboardInterrupt:
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING INTERRUPTED BY USER")
        logger.info(f"{'='*70}")

        # Save interrupted model
        interrupted_model_path = os.path.join(model_dir, "interrupted_model.zip")
        model.save(interrupted_model_path)
        logger.info(f"Interrupted model saved: {interrupted_model_path}")
        logger.info(f"{'='*70}\n")

    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error(f"ERROR DURING TRAINING")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {e}", exc_info=True)

        # Save error model for debugging
        error_model_path = os.path.join(model_dir, "error_model.zip")
        model.save(error_model_path)
        logger.error(f"Error model saved: {error_model_path}")
        logger.error(f"{'='*70}\n")
        raise

    finally:
        # Clean up
        env.close()
        eval_env.close()
        logger.info("Environments closed\n")

    return output_dir


# ============================================================
# Main Function
# ============================================================

# ============================================================
# Main Function
# ============================================================

def main():
    """Main training entry point."""

    parser = argparse.ArgumentParser(
        description="Train SAC agent for teleoperation with adaptive control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ============================================================
    # Environment Arguments
    # ============================================================
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument(
        "--model_path",
        type=str,
        # MODIFICATION: Removed 'required=True' and added a default path.
        default="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml",
        help="Path to MuJoCo XML model file"
    )
    env_group.add_argument(
        "--config",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5, 6],
        help="Delay configuration: 1=LOW, 2=MEDIUM, 3=HIGH, 4=NO_DELAY, 5=OBS_ONLY, 6=ACTION_ONLY"
    )
    env_group.add_argument(
        "--freq",
        type=int,
        default=500,
        help="Control frequency (Hz)"
    )
    env_group.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    env_group.add_argument(
        "--max_cartesian_error",
        type=float,
        default=0.3,
        help="Maximum allowed Cartesian error (m) before termination"
    )

    # ============================================================
    # Trajectory Arguments
    # ============================================================
    traj_group = parser.add_argument_group('Trajectory Configuration')
    traj_group.add_argument(
        "--trajectory_type",
        type=str,
        default="figure_8",
        choices=["figure_8", "square", "lissajous_complex"],
        help="Type of reference trajectory"
    )
    traj_group.add_argument(
        "--randomize_trajectory",
        action="store_true",
        help="Randomize trajectory parameters for better generalization"
    )

    # ============================================================
    # Controller Arguments
    # ============================================================
    ctrl_group = parser.add_argument_group('Controller Configuration')
    ctrl_group.add_argument(
        "--min_gain_ratio",
        type=float,
        default=0.3,
        help="Minimum gain ratio for adaptive PD (at high delay)"
    )
    ctrl_group.add_argument(
        "--delay_threshold",
        type=float,
        default=0.2,
        help="Delay threshold for gain adaptation (seconds)"
    )
    ctrl_group.add_argument(
        "--max_joint_change",
        type=float,
        default=0.1,
        help="Maximum joint change per IK solve (rad)"
    )
    ctrl_group.add_argument(
        "--continuity_gain",
        type=float,
        default=0.5,
        help="Null-space projection gain for trajectory continuity"
    )

    # ============================================================
    # Observation Arguments
    # ============================================================
    obs_group = parser.add_argument_group('Observation Configuration')
    obs_group.add_argument(
        "--target_history_len",
        type=int,
        default=10,
        help="Length of target position history for NN predictor"
    )
    obs_group.add_argument(
        "--action_history_len",
        type=int,
        default=5,
        help="Length of action history"
    )

    # ============================================================
    # Training Arguments
    # ============================================================
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument(
        "--output_path",
        type=str,
        default="./rl_training_output",
        help="Output directory for models and logs"
    )
    train_group.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Total training timesteps"
    )
    train_group.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    # ============================================================
    # SAC Arguments
    # ============================================================
    sac_group = parser.add_argument_group('SAC Hyperparameters')
    sac_group.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    sac_group.add_argument(
        "--buffer_size",
        type=int,
        default=100000,
        help="Replay buffer size"
    )
    sac_group.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training"
    )
    sac_group.add_argument(
        "--ent_coef",
        type=str,
        default="auto",
        help="Entropy coefficient ('auto' for automatic tuning)"
    )
    sac_group.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    sac_group.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Target network update rate"
    )
    sac_group.add_argument(
        "--learning_starts",
        type=int,
        default=20000,
        help="Steps before training starts"
    )
    sac_group.add_argument(
        "--train_freq",
        type=int,
        default=1,
        help="Training frequency (steps)"
    )
    sac_group.add_argument(
        "--gradient_steps",
        type=int,
        default=1,
        help="Gradient steps per training iteration"
    )
    sac_group.add_argument(
        "--target_update_interval",
        type=int,
        default=1,
        help="Target network update interval"
    )

    # ============================================================
    # Network Arguments
    # ============================================================
    net_group = parser.add_argument_group('Network Architecture')
    net_group.add_argument(
        "--net_arch",
        type=int,
        nargs='+',
        default=[256, 256],
        help="Network architecture (hidden layer sizes)"
    )
    net_group.add_argument(
        "--activation_fn",
        type=str,
        default="relu",
        choices=["relu", "tanh", "elu", "silu"],
        help="Activation function"
    )
    net_group.add_argument(
        "--use_custom_policy",
        action="store_true",
        help="Use custom policy with NN predictor"
    )

    # ============================================================
    # Callback Arguments
    # ============================================================
    callback_group = parser.add_argument_group('Callback Configuration')
    callback_group.add_argument(
        "--save_freq",
        type=int,
        default=25000,
        help="Model checkpoint frequency (steps)"
    )
    callback_group.add_argument(
        "--eval_freq",
        type=int,
        default=2500,
        help="Evaluation frequency (steps)"
    )
    callback_group.add_argument(
        "--eval_episodes",
        type=int,
        default=30,
        help="Number of evaluation episodes"
    )
    callback_group.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping"
    )
    callback_group.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (evaluations)"
    )
    callback_group.add_argument(
        "--min_improvement",
        type=float,
        default=1.0,
        help="Minimum reward improvement for early stopping"
    )

    # ============================================================
    # Policy Arguments
    # ============================================================
    policy_group = parser.add_argument_group('Policy Configuration')
    policy_group.add_argument(
        "--policy_type",
        type=str,
        default="learned_predictor",
        choices=["learned_predictor", "linear_interpolation", "default"],
        help="Type of policy: learned_predictor (NN), linear_interpolation (baseline), or default (no prediction)"
    )
    policy_group.add_argument(
        "--features_dim",
        type=int,
        default=256,
        help="Feature extractor output dimension"
    )
    policy_group.add_argument(
        "--predictor_arch",
        type=int,
        nargs='+',
        default=[128, 128, 64],
        help="Predictor network hidden layers"
    )
    policy_group.add_argument(
        "--controller_arch",
        type=int,
        nargs='+',
        default=[512],
        help="Controller feature extractor hidden layers"
    )

    # ============================================================
    # Parse Arguments
    # ============================================================
    args = parser.parse_args()

    # ============================================================
    # Process Arguments
    # ============================================================

    # Convert config number to ExperimentConfig enum
    config_map = {
        1: ExperimentConfig.LOW_DELAY,
        2: ExperimentConfig.MEDIUM_DELAY,
        3: ExperimentConfig.HIGH_DELAY,
        4: ExperimentConfig.NO_DELAY_BASELINE,
        5: ExperimentConfig.OBSERVATION_DELAY_ONLY,
        6: ExperimentConfig.ACTION_DELAY_ONLY,
    }
    args.config = config_map[args.config]

    # Convert trajectory type string to TrajectoryType enum
    trajectory_map = {
        "figure_8": TrajectoryType.FIGURE_8,
        "square": TrajectoryType.SQUARE,
        "lissajous_complex": TrajectoryType.LISSAJOUS_COMPLEX,
    }
    args.trajectory_type = trajectory_map[args.trajectory_type]

    # Convert activation function string to class
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU
    }
    args.activation_fn = activation_map[args.activation_fn]

    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)

    # ============================================================
    # Run Training
    # ============================================================
    try:
        output_dir = train_agent(args)
        logger.info(f"\nTraining completed successfully!")
        logger.info(f"Results saved to: {output_dir}\n")

    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}\n", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()