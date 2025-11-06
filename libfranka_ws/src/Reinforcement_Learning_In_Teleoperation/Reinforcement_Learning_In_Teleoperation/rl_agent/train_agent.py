"""
The main training script, the primary entry point for training a Recurrent-PPO agent
"""

import os
import sys
from datetime import datetime
import logging
import argparse
import torch
import numpy as np

# stable-baselines3 imports
from stable_baselines3.common.vec_env import  SubprocVecEnv  # For parallel envs
from stable_baselines3.common.env_util import make_vec_env  # To create vec envs

# Custom imports
from Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
from Reinforcement_Learning_In_Teleoperation.rl_agent.ppo_training_algorithm import RecurrentPPOTrainer
from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    PPO_TOTAL_TIMESTEPS,
    CHECKPOINT_DIR,
    NUM_ENVIRONMENTS
)


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to file and console."""
    
    log_file = os.path.join(output_dir, "training.log")
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - detailed and verbose
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG) 
    console_handler.setFormatter(detailed_formatter) 
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()  # Remove existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger(__name__)
    
    return logger


def validate_environment(env: TeleoperationEnvWithDelay, logger: logging.Logger) -> bool:
    """Checking Custom Environment"""
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
    """Main training function: sets up environment, trainer, and runs training loop."""
    
    # Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = args.config.name
    trajectory_name = args.trajectory_type.value
    
    # File name
    run_name = f"RecPPO_{config_name}_{trajectory_name}_{timestamp}"
        
    # Determine base output directory
    base_output_dir = CHECKPOINT_DIR
    output_dir = os.path.join(base_output_dir, run_name)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Failed to create output directory {output_dir}: {e}")
        sys.exit(1)
    
    # Setup Logging
    logger = setup_logging(output_dir)
    logger.info(f"Run Name: {run_name}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("Training Configuration:")
    logger.info(f"  Delay Config: {args.config.name}")
    logger.info(f"  Trajectory Type: {args.trajectory_type.value}")
    logger.info(f"  Randomize Trajectory: {args.randomize_trajectory}")
    logger.info(f"  Total Timesteps: {args.timesteps:,}")
    logger.info(f"  Random Seed: {args.seed if args.seed is not None else 'None (random)'}")
    logger.info("")
    
    # Set Random Seeds
    if args.seed is not None:
        logger.info(f"Setting random seed: {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info("Random seeds set for NumPy, PyTorch, and CUDA")
        logger.info("")

    # Environment setup
    N_env = NUM_ENVIRONMENTS 
    
    env = None
    try:
        # Define a function that creates a single environment instance
        # This is needed by make_vec_env
        def make_env():
            env_instance = TeleoperationEnvWithDelay(
                delay_config=args.config,
                trajectory_type=args.trajectory_type,
                randomize_trajectory=args.randomize_trajectory,
                render_mode=args.render
            )
           
            return env_instance

        # Create the vectorized environment
        env = make_vec_env(
            make_env,
            n_envs=N_env,
            seed=args.seed, # Handles seeding each sub-environment
            vec_env_cls=SubprocVecEnv # for parallparallelism
        )

        logger.info(f"  Vectorized Environment: {env.__class__.__name__}")
        logger.info(f"  Number of Envs: {env.num_envs}")
        logger.info(f"  Observation Space: {env.observation_space.shape}") # Shape is usually same
        logger.info(f"  Action Space: {env.action_space.shape}") # Shape is usually same
        logger.info("")

    except Exception as e:
        logger.error(f"Failed to create vectorized environment: {e}", exc_info=True)
        if env:
            env.close()
        sys.exit(1)
   
    # Trrainer Initialization
    logger.info("Initializing Recurrent-PPO trainer...")
    trainer = None
    
    try:
        trainer = RecurrentPPOTrainer(env=env)
        trainer.checkpoint_dir = output_dir
        
        logger.info(f"  Trainer: RecurrentPPOTrainer")
        logger.info(f"  Policy Parameters: {trainer.policy.count_parameters():,}")
        logger.info(f"  Checkpoint Directory: {output_dir}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}", exc_info=True)
        env.close()
        sys.exit(1)
   
    # Start Training Loop
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
        # clean up the environment
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
            logger.info("Training completed successfully!")
        else:
            logger.info("Training ended prematurely.")

def parse_arguments() -> argparse.Namespace:
    """Make parse arguments."""
    
    parser = argparse.ArgumentParser(
        description="Train an end-to-end Recurrent-PPO agent for delayed teleoperation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment configuration group
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument("--config", type=str, default="2", choices=['1', '2', '3', '4'], help="Delay configuration preset (1=LOW, 2=MEDIUM, 3=HIGH, 4=EXTREME)")
    exp_group.add_argument("--trajectory-type", type=str.lower, default="figure_8", choices=[t.value for t in TrajectoryType], help="Reference trajectory type (figure_8, square, lissajous_complex)")
    exp_group.add_argument("--randomize-trajectory", action="store_true", help="Randomize trajectory parameters during training for better generalization")
    exp_group.add_argument("--timesteps", type=int, default=PPO_TOTAL_TIMESTEPS, help="Total training timesteps")
    exp_group.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (None for random seed)")
    exp_group.add_argument("--render", type=str.lower, default= 'human', choices=['human', 'rgb_array', 'none'], help="Rendering mode for visualization ('human' opens a live plot).")
    
    args = parser.parse_args()
    
    # Get all enum members from ExperimentConfig
    config_options = list(ExperimentConfig)
    
    if len(config_options) < 4:
        raise ValueError(f"Config mapping needs 4 options, but ExperimentConfig only has {len(config_options)}")
        
    # Create a mapping from string numbers to the first 4 enum members
    CONFIG_MAP = {
        '1': config_options[0], # e.g., LOW_DELAY
        '2': config_options[1], # e.g., MEDIUM_DELAY
        '3': config_options[2], # e.g., HIGH_DELAY
        '4': config_options[3]  # e.g., EXTREME_DELAY
    }
    
    # Convert the string '1', '2', etc., to the actual enum object
    args.config = CONFIG_MAP[args.config]
        
        
    # Convert trajectory type string to TrajectoryType enum
    args.trajectory_type = next(
        t for t in TrajectoryType 
        if t.value.lower() == args.trajectory_type.lower()
    )
   
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