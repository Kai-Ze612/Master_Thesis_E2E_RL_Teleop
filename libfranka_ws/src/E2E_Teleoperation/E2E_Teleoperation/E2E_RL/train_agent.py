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
from E2E_Teleoperation.E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnvWithDelay
from E2E_Teleoperation.E2E_Teleoperation.E2E_RL.local_robot_simulator import TrajectoryType
from E2E_Teleoperation.E2E_Teleoperation.E2E_RL.sac_training_algorithm import SACTrainer
from E2E_Teleoperation.E2E_Teleoperation.utils.delay_simulator import ExperimentConfig
import E2E_Teleoperation.E2E_Teleoperation.config.robot_config as cfg


def setup_logging(output_dir: str) -> logging.Logger:
    log_file = os.path.join(output_dir, "training.log")
    detailed_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG) 
    console_handler.setFormatter(detailed_formatter) 
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return logging.getLogger(__name__)

def train_agent(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_name = f"E2E_SAC_{args.config.name}_{args.trajectory_type.value}_{timestamp}"
    
    output_dir = os.path.join(cfg.CHECKPOINT_DIR_RL, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("Training Configuration (Joint End-to-End):")
    logger.info(f"  Delay: {args.config.name}")
    logger.info(f"  Trajectory: {args.trajectory_type.value}")
    logger.info(f"  Total Timesteps: {args.timesteps}")
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    logger.info("Creating vectorized environment...")
    env = None
    try:
        def make_env():
            return TeleoperationEnvWithDelay(
                delay_config=args.config,
                trajectory_type=args.trajectory_type,
                randomize_trajectory=args.randomize_trajectory,
                render_mode=args.render,
                # [MODIFICATION] No lstm_model_path needed
                lstm_model_path=None 
            )

        env = make_vec_env(make_env, n_envs=cfg.NUM_ENVIRONMENTS, seed=args.seed, vec_env_cls=SubprocVecEnv)

    except Exception as e:
        logger.error(f"Failed to create env: {e}", exc_info=True)
        if env: env.close()
        sys.exit(1)
   
    logger.info("Initializing SACTrainer...")
    trainer = None
    try:
        trainer = SACTrainer(
            env=env,
            val_delay_config=args.config)
        trainer.checkpoint_dir = output_dir
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}", exc_info=True)
        env.close()
        sys.exit(1)
   
    try:
        trainer.train(total_timesteps=args.timesteps)
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        trainer.save_checkpoint("interrupted_policy.pth")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        trainer.save_checkpoint("crash_policy.pth")
    finally:
        if env: env.close()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_group = parser.add_argument_group('Experiment Configuration')
    # Config 1=Low, 2=High, 3=HighVariance
    exp_group.add_argument("--config", type=str, default="3", choices=['1', '2', '3'])
    exp_group.add_argument("--trajectory-type", type=str.lower, default="figure_8", choices=[t.value for t in TrajectoryType])
    exp_group.add_argument("--randomize-trajectory", action="store_true")
    
    # [MODIFICATION] Use cfg.SAC_TOTAL_TIMESTEPS
    exp_group.add_argument("--timesteps", type=int, default=cfg.SAC_TOTAL_TIMESTEPS)
    exp_group.add_argument("--seed", type=int, default=None)
    exp_group.add_argument("--render", type=str.lower, default=None, choices=['human', 'rgb_array', 'none'])
    
    args = parser.parse_args()
    config_options = list(ExperimentConfig)
    # Map input '1','2','3' to the Enum values
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2]}
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    return args

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    train_agent(parse_arguments())