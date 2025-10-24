# File: train_recurrent_ppo.py
# (Place this in a location like Reinforcement_Learning_In_Teleoperation/ or Reinforcement_Learning_In_Teleoperation/rl_agent/)

"""
Main training script for end-to-end Recurrent-PPO agent
for joint-space teleoperation with delay compensation.

Usage:
    python train_recurrent_ppo.py [--config CONFIG_NAME] [--trajectory-type TRAJ_NAME] [...]
"""

# Python imports
import os
import sys
from datetime import datetime
import logging
import argparse
import torch
import numpy as np

# --- Ensure project modules can be imported ---
# Adjust the path based on where you place this script relative to your project root
script_dir = os.path.dirname(os.path.abspath(__file__))
# Example: If script is in project_root/rl_agent/, navigate up one level
project_root = os.path.dirname(script_dir)
# Example: If script is in project_root/, use script_dir directly
# project_root = script_dir
if project_root not in sys.path:
    sys.path.append(project_root)
# --- Custom imports ---
# Environment (Adjust path based on your structure)
from Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
# Delay and Trajectory Types (Adjust path based on your structure)
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
# Algorithm Trainer (Adjust path based on your structure)
from Reinforcement_Learning_In_Teleoperation.Reinforcement_Learning_In_Teleoperation.rl_agent.ppo_training_algorithm import RecurrentPPOTrainer
# Config (Imports hyperparameters directly)
from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    PPO_TOTAL_TIMESTEPS, CHECKPOINT_DIR, DEFAULT_CONTROL_FREQ
)


# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to console
        # Optional: Add FileHandler to log to a file in the output directory
        # logging.FileHandler(os.path.join(output_dir, "training.log"))
    ]
)
logger = logging.getLogger(__name__) # Get logger for this script

def train_agent(args: argparse.Namespace) -> None:
    """Sets up the environment, initializes the trainer, and runs the training loop."""

    # --- Setup Output Directories ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = args.config.name # Use Enum name
    trajectory_name = args.trajectory_type.value # Use Enum value (string)
    run_tag = f"_{args.tag}" if args.tag else ""
    # Define a clear run name including the algorithm
    run_name = f"RecPPO_{config_name}_{trajectory_name}{run_tag}_{timestamp}"

    # Use CHECKPOINT_DIR from config as the base output path, handle None case
    base_output_dir = args.output_path or CHECKPOINT_DIR or "./rl_training_output/recurrent_ppo" # Default path
    output_dir = os.path.join(base_output_dir, run_name)

    try:
        os.makedirs(output_dir, exist_ok=True) # Trainer will save models inside this directory
        logger.info(f"Output directory: {output_dir}")
        # Log arguments used for this run
        logger.info("--- Run Arguments ---")
        for arg, value in vars(args).items():
            # Log Enum names for clarity
            if isinstance(value, (ExperimentConfig, TrajectoryType)):
                 logger.info(f"  --{arg}: {value.name}")
            else:
                 logger.info(f"  --{arg}: {value}")
        logger.info("---------------------")
    except OSError as e:
        logger.error(f"Failed to create output directories: {e}")
        return

    # --- Device Setup ---
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        try:
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            logger.error(f"Could not get GPU name: {e}")


    # --- Environment Setup ---
    logger.info("Creating training environment...")
    try:
        # Pass the selected config, trajectory, seed etc.
        env = TeleoperationEnvWithDelay(
            experiment_config=args.config,
            trajectory_type=args.trajectory_type,
            randomize_trajectory=args.randomize_trajectory,
            seed=args.seed
        )
        logger.info(f"Environment created: {env.delay_simulator.config_name}, Freq: {env.control_freq} Hz")
        # Validate frequency if desired
        if env.control_freq != DEFAULT_CONTROL_FREQ:
             logger.warning(f"Env freq ({env.control_freq}Hz) != Config default ({DEFAULT_CONTROL_FREQ}Hz)")

    except Exception as e:
        logger.error(f"Failed to create environment: {e}", exc_info=True)
        return

    # --- Trainer Initialization ---
    logger.info("Initializing Recurrent-PPO trainer...")
    try:
        # Pass the environment instance and device
        # The trainer will use hyperparameters imported from config.py
        trainer = RecurrentPPOTrainer(env, device=device)
        # Set the specific directory for this run's checkpoints within the trainer
        trainer.checkpoint_dir = output_dir # Trainer will save inside this directory
        logger.info("Trainer initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}", exc_info=True)
        env.close()
        return

    # --- Start Training ---
    try:
        logger.info(f"Starting training for {args.timesteps:,} timesteps...")
        # Train using the total timesteps specified (command line or config default)
        trainer.train(total_timesteps=args.timesteps)
        logger.info("Training finished.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving final model...")
        # Save model gracefully on interrupt
        final_path = os.path.join(trainer.checkpoint_dir, "interrupted_final_policy.pth")
        try:
            trainer.policy.save(final_path)
            logger.info(f"Interrupted model saved to {final_path}")
        except Exception as save_e:
            logger.error(f"Could not save interrupted model: {save_e}")
    except Exception as e:
         logger.error(f"An error occurred during training: {e}", exc_info=True)
         # Attempt to save a crash model for debugging
         crash_path = os.path.join(trainer.checkpoint_dir, "crash_policy.pth")
         try: # Try saving, might fail if error is severe
             trainer.policy.save(crash_path)
             logger.info(f"Saved crash model to {crash_path}")
         except Exception as save_e:
              logger.error(f"Could not save crash model: {save_e}")
    finally:
        # --- Cleanup ---
        env.close()
        logger.info("Environment closed.")

def main():
    """Parses command-line arguments and launches the Recurrent-PPO training."""
    parser = argparse.ArgumentParser(
        description="Train an end-to-end Recurrent-PPO agent for teleoperation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Experiment Configuration ---
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument(
        "--config", type=str.upper, default="MEDIUM_DELAY", # Default to Medium Delay, use upper case
        choices=[e.name for e in ExperimentConfig], # Use Enum names
        help="Delay config preset name from ExperimentConfig (e.g., LOW_DELAY, MEDIUM_DELAY, OBSERVATION_DELAY_ONLY)."
    )
    exp_group.add_argument(
        "--trajectory-type", type=str.lower, default="figure_8",
        choices=[t.value for t in TrajectoryType], # Use Enum values (strings)
        help="Type of reference trajectory (e.g., figure_8, square)."
    )
    exp_group.add_argument(
        "--randomize-trajectory", action="store_true",
        help="If set, randomize trajectory parameters during training."
    )
    exp_group.add_argument(
        "--timesteps", type=int, default=PPO_TOTAL_TIMESTEPS, # Default from config
        help="Total training timesteps."
    )
    exp_group.add_argument(
        "--output-path", type=str, default=None, # Default based on CHECKPOINT_DIR
        help=f"Base directory to save models and logs (defaults to config's CHECKPOINT_DIR or ./rl_training_output/recurrent_ppo)."
    )
    exp_group.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (numpy, torch, env)."
    )
    exp_group.add_argument(
        "--tag", type=str, default=None,
        help="Optional tag to append to the run name for identification."
    )
    exp_group.add_argument(
        "--device", type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
        help="Device for training ('auto' uses cuda if available, else cpu)."
    )

    # --- No algorithm hyperparameters here - they are in config.py ---

    args = parser.parse_args()

    # --- Process Arguments ---
    # Convert string names from argparse back to Enum members
    try:
        # Match case-insensitively just in case
        args.config = ExperimentConfig[args.config.upper()]
    except KeyError:
        logger.error(f"Invalid --config name provided: '{args.config}'. Available: {[e.name for e in ExperimentConfig]}")
        sys.exit(1)

    try:
        # Match case-insensitively
        args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    except StopIteration:
        logger.error(f"Invalid --trajectory-type value provided: '{args.trajectory_type}'. Available: {[t.value for t in TrajectoryType]}")
        sys.exit(1)

    # Set seed for reproducibility if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            # Consider adding these for full determinism, but they can slow down training
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        logger.info(f"Using random seed: {args.seed}")

    # --- Launch Training ---
    train_agent(args)


if __name__ == "__main__":
    main()