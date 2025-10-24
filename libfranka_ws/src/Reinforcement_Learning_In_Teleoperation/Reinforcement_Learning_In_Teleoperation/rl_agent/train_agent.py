"""
Main training script for end-to-end Recurrent-PPO agent
for joint-space teleoperation with delay compensation.

This script orchestrates the training process:
    1. Parse command-line arguments
    2. Setup output directories and logging
    3. Create training environment
    4. Initialize Recurrent-PPO trainer
    5. Run training loop with checkpointing
    6. Handle graceful shutdown and error recovery

Usage:
    # Train with medium delay and figure-8 trajectory (default)
    python train_agent.py
    
    # Train with high delay and square trajectory
    python train_agent.py --config HIGH_DELAY --trajectory-type square
    
    # Train with custom timesteps and seed
    python train_agent.py --timesteps 5000000 --seed 42
    
    # Train with trajectory randomization
    python train_agent.py --randomize-trajectory --tag "rand_traj"

Author: [Your Name]
Date: [Date]
"""

import os
import sys
from datetime import datetime
import logging
import argparse
from typing import Optional
import torch
import numpy as np

# --- Ensure project modules can be imported ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Custom imports ---
from Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
from Reinforcement_Learning_In_Teleoperation.rl_agent.ppo_training_algorithm import RecurrentPPOTrainer
from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    PPO_TOTAL_TIMESTEPS,
    CHECKPOINT_DIR,
    DEFAULT_CONTROL_FREQ
)


def setup_logging(output_dir: str, console_level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Args:
        output_dir: Directory to save log file
        console_level: Logging level for console output
        
    Returns:
        Configured logger instance
    """
    log_file = os.path.join(output_dir, "training.log")
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simpler, less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def validate_environment(env: TeleoperationEnvWithDelay, logger: logging.Logger) -> bool:
    """
    Perform basic validation checks on the environment.
    
    Args:
        env: Environment instance to validate
        logger: Logger for outputting messages
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check observation space
        obs, info = env.reset()
        expected_obs_shape = env.observation_space.shape
        if obs.shape != expected_obs_shape:
            logger.error(f"Observation shape mismatch: got {obs.shape}, expected {expected_obs_shape}")
            return False
        
        # Check action space
        expected_action_shape = env.action_space.shape
        dummy_action = np.zeros(expected_action_shape)
        
        # Try a step
        next_obs, reward, terminated, truncated, info = env.step(dummy_action)
        
        if next_obs.shape != expected_obs_shape:
            logger.error(f"Next observation shape mismatch: got {next_obs.shape}, expected {expected_obs_shape}")
            return False
        
        logger.info("Environment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}", exc_info=True)
        return False


def train_agent(args: argparse.Namespace) -> None:
    """
    Main training function: sets up environment, trainer, and runs training loop.
    
    Args:
        args: Parsed command-line arguments
    """
    
    # --- Setup Output Directories ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = args.config.name
    trajectory_name = args.trajectory_type.value
    run_tag = f"_{args.tag}" if args.tag else ""
    
    # Create descriptive run name
    run_name = f"RecPPO_{config_name}_{trajectory_name}{run_tag}_{timestamp}"
    
    # Determine base output directory
    base_output_dir = args.output_path or CHECKPOINT_DIR or "./rl_training_output/recurrent_ppo"
    output_dir = os.path.join(base_output_dir, run_name)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Failed to create output directory {output_dir}: {e}")
        sys.exit(1)
    
    # --- Setup Logging ---
    logger = setup_logging(output_dir)
    
    logger.info("="*70)
    logger.info("Recurrent-PPO Training for Delayed Teleoperation")
    logger.info("="*70)
    logger.info(f"Run Name: {run_name}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("")
    
    # --- Log Arguments ---
    logger.info("Training Configuration:")
    logger.info(f"  Delay Config: {args.config.name}")
    logger.info(f"  Trajectory Type: {args.trajectory_type.value}")
    logger.info(f"  Randomize Trajectory: {args.randomize_trajectory}")
    logger.info(f"  Total Timesteps: {args.timesteps:,}")
    logger.info(f"  Random Seed: {args.seed if args.seed is not None else 'None (random)'}")
    logger.info(f"  Device: {args.device}")
    if args.tag:
        logger.info(f"  Tag: {args.tag}")
    logger.info("")
    
    # --- Device Setup ---
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Device: {device}")
    if device == 'cuda':
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  GPU: {gpu_name}")
            logger.info(f"  GPU Memory: {gpu_memory:.1f} GB")
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
    logger.info("")
    
    # --- Set Random Seeds ---
    if args.seed is not None:
        logger.info(f"Setting random seed: {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            # For full determinism (may slow down training):
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        logger.info("Random seeds set for NumPy, PyTorch, and CUDA")
        logger.info("")
    
    # --- Environment Setup ---
    logger.info("Creating training environment...")
    env = None
    try:
        env = TeleoperationEnvWithDelay(
            delay_config=args.config,  # Fixed: was experiment_config
            trajectory_type=args.trajectory_type,
            randomize_trajectory=args.randomize_trajectory,
            seed=args.seed
        )
        
        logger.info(f"  Environment: TeleoperationEnvWithDelay")
        logger.info(f"  Delay Config: {env.delay_simulator.config_name}")
        logger.info(f"  Control Frequency: {env.control_freq} Hz")
        logger.info(f"  Max Episode Steps: {env.max_episode_steps}")
        logger.info(f"  Observation Space: {env.observation_space.shape}")
        logger.info(f"  Action Space: {env.action_space.shape}")
        
        # Validate frequency matches config
        if env.control_freq != DEFAULT_CONTROL_FREQ:
            logger.warning(f"Environment frequency ({env.control_freq} Hz) differs from "
                         f"config default ({DEFAULT_CONTROL_FREQ} Hz)")
        
        # Validate environment
        logger.info("")
        logger.info("Validating environment...")
        if not validate_environment(env, logger):
            logger.error("Environment validation failed. Aborting training.")
            if env:
                env.close()
            sys.exit(1)
        logger.info("")
        
    except Exception as e:
        logger.error(f"Failed to create environment: {e}", exc_info=True)
        if env:
            env.close()
        sys.exit(1)
    
    # --- Trainer Initialization ---
    logger.info("Initializing Recurrent-PPO trainer...")
    trainer = None
    try:
        trainer = RecurrentPPOTrainer(env=env, device=device)
        
        # Override checkpoint directory for this specific run
        trainer.checkpoint_dir = output_dir
        
        logger.info(f"  Trainer: RecurrentPPOTrainer")
        logger.info(f"  Policy Parameters: {trainer.policy.count_parameters():,}")
        logger.info(f"  Checkpoint Directory: {output_dir}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}", exc_info=True)
        env.close()
        sys.exit(1)
    
    # --- Start Training ---
    training_successful = False
    try:
        logger.info("="*70)
        logger.info("Starting Training")
        logger.info("="*70)
        logger.info("")
        
        trainer.train(total_timesteps=args.timesteps)
        
        logger.info("")
        logger.info("="*70)
        logger.info("Training Completed Successfully")
        logger.info("="*70)
        training_successful = True
        
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("="*70)
        logger.warning("Training Interrupted by User")
        logger.warning("="*70)
        logger.warning("Saving interrupted model...")
        
        # Save model gracefully on interrupt
        interrupt_path = os.path.join(trainer.checkpoint_dir, "interrupted_policy.pth")
        try:
            trainer.policy.save(interrupt_path)
            logger.info(f"Interrupted model saved to: {interrupt_path}")
        except Exception as save_e:
            logger.error(f"Could not save interrupted model: {save_e}")
    
    except Exception as e:
        logger.error("")
        logger.error("="*70)
        logger.error("Training Failed with Error")
        logger.error("="*70)
        logger.error(f"Error: {e}", exc_info=True)
        
        # Attempt to save crash model for debugging
        crash_path = os.path.join(trainer.checkpoint_dir, "crash_policy.pth")
        try:
            trainer.policy.save(crash_path)
            logger.info(f"Crash model saved to: {crash_path}")
        except Exception as save_e:
            logger.error(f"Could not save crash model: {save_e}")
    
    finally:
        # --- Cleanup ---
        logger.info("")
        logger.info("Cleaning up...")
        if env:
            env.close()
            logger.info("Environment closed")
        
        logger.info("")
        logger.info("="*70)
        logger.info(f"Output Directory: {output_dir}")
        logger.info("="*70)
        
        if training_successful:
            logger.info("Training completed successfully! 🎉")
        else:
            logger.info("Training ended prematurely.")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train an end-to-end Recurrent-PPO agent for delayed teleoperation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Experiment Configuration ---
    exp_group = parser.add_argument_group('Experiment Configuration')
    
    exp_group.add_argument(
        "--config",
        type=str.upper,
        default="MEDIUM_DELAY",
        choices=[e.name for e in ExperimentConfig],
        help="Delay configuration preset (LOW_DELAY, MEDIUM_DELAY, HIGH_DELAY, etc.)"
    )
    
    exp_group.add_argument(
        "--trajectory-type",
        type=str.lower,
        default="figure_8",
        choices=[t.value for t in TrajectoryType],
        help="Reference trajectory type (figure_8, square, lissajous_complex)"
    )
    
    exp_group.add_argument(
        "--randomize-trajectory",
        action="store_true",
        help="Randomize trajectory parameters during training for better generalization"
    )
    
    exp_group.add_argument(
        "--timesteps",
        type=int,
        default=PPO_TOTAL_TIMESTEPS,
        help="Total training timesteps"
    )
    
    exp_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (None for random seed)"
    )
    
    exp_group.add_argument(
        "--device",
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help="Device for training (auto selects CUDA if available)"
    )
    
    # --- Output Configuration ---
    output_group = parser.add_argument_group('Output Configuration')
    
    output_group.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=f"Base directory for saving models and logs (default: {CHECKPOINT_DIR or './rl_training_output/recurrent_ppo'})"
    )
    
    output_group.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag to append to run name for easy identification"
    )
    
    args = parser.parse_args()
    
    # --- Process and Validate Arguments ---
    try:
        args.config = ExperimentConfig[args.config.upper()]
    except KeyError:
        print(f"ERROR: Invalid --config '{args.config}'")
        print(f"Available options: {[e.name for e in ExperimentConfig]}")
        sys.exit(1)
    
    try:
        args.trajectory_type = next(
            t for t in TrajectoryType 
            if t.value.lower() == args.trajectory_type.lower()
        )
    except StopIteration:
        print(f"ERROR: Invalid --trajectory-type '{args.trajectory_type}'")
        print(f"Available options: {[t.value for t in TrajectoryType]}")
        sys.exit(1)
    
    # Validate timesteps
    if args.timesteps <= 0:
        print(f"ERROR: --timesteps must be positive, got {args.timesteps}")
        sys.exit(1)
    
    return args


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        train_agent(args)
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()