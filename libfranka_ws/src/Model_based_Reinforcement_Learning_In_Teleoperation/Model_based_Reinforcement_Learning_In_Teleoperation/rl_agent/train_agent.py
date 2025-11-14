"""
The main training script for Model-Based Reinforcement Learning (LSTM + SAC) agent
"""

import os
import sys
from datetime import datetime
import logging
import torch
import numpy as np
import argparse
import multiprocessing

# stable-baselines3 imports
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Custom imports
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_training_algorithm import SACTrainer
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    SAC_TOTAL_TIMESTEPS,
    CHECKPOINT_DIR_RL,
    NUM_ENVIRONMENTS,
    OBS_DIM,
    LSTM_MODEL_PATH,
)


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to file and console."""
    
    log_file = os.path.join(output_dir, "training.log")
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-m-%d %H:%M:%S'
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
    """Validate Custom Environment."""
    try:
        # Check observation space
        obs, info = env.reset()
        expected_obs_shape = env.observation_space.shape
        if obs.shape != expected_obs_shape:
            logger.error(f"Observation shape mismatch: got {obs.shape}, expected {expected_obs_shape}")
            return False
        
        # NEW: Verify observation dimension is exactly 112D
        if obs.shape[0] != 112:
            logger.error(f"Observation dimension should be 112D, got {obs.shape[0]}D")
            return False
        
        # Check action space
        expected_action_shape = env.action_space.shape
        dummy_action = np.zeros(expected_action_shape)
        
        # Try a step
        next_obs, reward, terminated, truncated, info = env.step(dummy_action)
        
        if next_obs.shape != expected_obs_shape:
            logger.error(f"Next observation shape mismatch: got {next_obs.shape}, expected {expected_obs_shape}")
            return False
        
        logger.info("Environment validation passed ✓")
        logger.info(f"  Observation dimension: {obs.shape[0]}D")
        logger.info(f"  Action dimension: {env.action_space.shape}")
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
    
    run_name = f"ModelBasedSAC_{config_name}_{trajectory_name}_{timestamp}"
    
    # Output directory
    base_output_dir = CHECKPOINT_DIR_RL
    output_dir = os.path.join(base_output_dir, run_name)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Failed to create output directory {output_dir}: {e}")
        sys.exit(1)
    
    # Setup Logging
    logger = setup_logging(output_dir)
    logger.info("Training Configuration:")
    logger.info(f"  Delay Config: {args.config.name}")
    logger.info(f"  Trajectory Type: {args.trajectory_type.value}")
    logger.info(f"  Randomize Trajectory: {args.randomize_trajectory}")
    logger.info(f"  Total Timesteps: {args.timesteps:,}")
    logger.info(f"  Random Seed: {args.seed if args.seed is not None else 'None (random)'}")
    logger.info("")
    
    # Set Random Seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Environment setup
    N_env = NUM_ENVIRONMENTS 
    
    # Create Vectorized Environment
    logger.info("Creating vectorized environment...")
    env = None
    try:
        def make_env():
            env_instance = TeleoperationEnvWithDelay(
                delay_config=args.config,
                trajectory_type=args.trajectory_type,
                randomize_trajectory=args.randomize_trajectory,
                render_mode=args.render
            )
            return env_instance

        env = make_vec_env(
            make_env,
            n_envs=N_env,
            seed=args.seed,
            vec_env_cls=SubprocVecEnv
        )

        logger.info("Observation Structure (112D):")
        logger.info(f"  - Remote state: 14D (position 7D + velocity 7D)")
        logger.info(f"  - Remote history: 70D (5 timesteps × 14D)")
        logger.info(f"  - LSTM prediction: 14D (position 7D + velocity 7D)")
        logger.info(f"  - Current error: 14D (position error 7D + velocity error 7D)")
        logger.info(f"  - Total: {OBS_DIM}D")
        logger.info("")

    except Exception as e:
        logger.error(f"Failed to create vectorized environment: {e}", exc_info=True)
        if env:
            env.close()
        sys.exit(1)
   
    # Trainer Initialization
    logger.info("Initializing Model-Based SAC trainer...")
    trainer = None
    lstm_model_path = LSTM_MODEL_PATH
    
    # check LSTM model path
    if not os.path.isfile(lstm_model_path):
        logger.error(f"LSTM model file not found at: {lstm_model_path}")
        env.close()
        sys.exit(1)
    
    try:
        trainer = SACTrainer(
            env=env,
            pretrained_estimator_path=lstm_model_path
        )
        trainer.checkpoint_dir = output_dir  # Pass the run-specific dir
        
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
        
        trainer.save_checkpoint("interrupted_policy.pth")
        logger.info(f"Interrupted model saved to: {trainer.checkpoint_dir}")
    
    except Exception as e:
        logger.error("")
        logger.error("="*70)
        logger.error("Training Failed with Error")
        logger.error("="*70)
        logger.error(f"Error: {e}", exc_info=True)
        
        trainer.save_checkpoint("crash_policy.pth")
        logger.info(f"Crash model saved to: {trainer.checkpoint_dir}")
    
    finally:
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
        description="Train a Model-Based (LSTM+SAC) agent for delayed teleoperation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment configuration group
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument("--config", type=str, default="2", choices=['1', '2', '3', '4'], help="Delay configuration preset (1=LOW, 2=MEDIUM, 3=HIGH, 4=EXTREME)")
    exp_group.add_argument("--trajectory-type", type=str.lower, default="figure_8", choices=[t.value for t in TrajectoryType], help="Reference trajectory type (figure_8, square, lissajous_complex)")
    exp_group.add_argument("--randomize-trajectory", action="store_true", help="Randomize trajectory parameters during training for better generalization")
    exp_group.add_argument("--timesteps", type=int, default=SAC_TOTAL_TIMESTEPS, help="Total training timesteps")
    exp_group.add_argument("--seed", type=int, default=None, help="Random reproducibility seed (None for random seed)")
    exp_group.add_argument("--render", type=str.lower, default=None, choices=['human', 'rgb_array', 'none'], help="Rendering mode for visualization ('human' opens a live plot).")
    
    args = parser.parse_args()
    
    # Map config string to ExperimentConfig enum
    config_options = list(ExperimentConfig)
    if len(config_options) < 4:
        raise ValueError(f"Config mapping needs 4 options, but ExperimentConfig only has {len(config_options)}")
        
    CONFIG_MAP = {
        '1': config_options[0], # e.g., LOW_DELAY
        '2': config_options[1], # e.g., MEDIUM_DELAY
        '3': config_options[2], # e.g., HIGH_DELAY
        '4': config_options[3]  # e.g., FULL_RANGE_DELAY
    }
    args.config = CONFIG_MAP[args.config]
    
    # Map trajectory type string to TrajectoryType enum
    args.trajectory_type = next(
        t for t in TrajectoryType 
        if t.value.lower() == args.trajectory_type.lower()
    )
   
    if args.timesteps <= 0:
        print(f"ERROR: --timesteps must be positive, got {args.timesteps}")
        sys.exit(1)
    
    return args


def main():
    """Main entry point."""
    multiprocessing.set_start_method('spawn', force=True)
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