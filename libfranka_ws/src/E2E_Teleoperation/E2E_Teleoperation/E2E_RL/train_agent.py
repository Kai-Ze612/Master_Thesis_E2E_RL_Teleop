"""
The main training script for E2E Reinforcement Learning (LSTM + SAC) agent
"""

import os
import sys
from datetime import datetime
import logging
import torch
import numpy as np
import argparse
import multiprocessing
from pathlib import Path 

from E2E_Teleoperation.config import robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig
from E2E_Teleoperation.E2E_RL.local_robot_simulator import TrajectoryType
from E2E_Teleoperation.E2E_RL.sac_training_algorithm import Trainer

def setup_logging(output_dir: Path):
    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def train_agent(args: argparse.Namespace) -> None:
    # 1. Setup Output Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"E2E_RL_{args.config.name}_{args.trajectory_type.value}_{timestamp}"
    
    output_dir = cfg.CHECKPOINT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info(f"STARTING TRAINING: {run_name}")
    logger.info(f"Checkpoint Dir: {output_dir}")
    logger.info(f"Model Path: {cfg.DEFAULT_MUJOCO_MODEL_PATH}")
    logger.info("=" * 60)
    
    # 2. Seeding
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # 3. Environment
    logger.info("Initializing Environment...")
    render_mode = "human" if args.render else None
    
    env = None
    try:
        env = TeleoperationEnv(
            delay_config=args.config,
            seed=args.seed,
            render_mode=render_mode
        )
    except FileNotFoundError as e:
        logger.error(f"Environment Error: {e}")
        logger.error("Please check DEFAULT_MUJOCO_MODEL_PATH in robot_config.py")
        sys.exit(1)
    
    # 4. Trainer
    trainer = Trainer(env, str(output_dir))
    
    try:
        # Stage 1: LSTM Encoder Pre-training
        if not args.skip_stage1:
            trainer.train_stage_1_encoder()
        else:
            logger.info("Skipping Stage 1 (Encoder Pre-training)...")
        
        # Stage 2: BC with Recovery Learning
        trainer.train_stage_2_with_recovery()
        
        # Stage 3: SAC Fine-tuning (optional)
        if args.stage3:
            trainer.train_stage_3_sac()
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        trainer.save_checkpoint("interrupted_model.pth")
    except Exception as e:
        logger.error(f"Training crashed: {e}", exc_info=True)
        trainer.save_checkpoint("crash_model.pth")
    finally:
        trainer.save_checkpoint("final_model.pth")
        if env: 
            env.close()
        logger.info("Training Finished.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2E Training with Recovery Learning", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument(
        "--config", type=str, default="3", choices=['1', '2', '3'], 
        help="Delay config (1=Low, 2=High, 3=Var)"
    )
    exp_group.add_argument(
        "--trajectory-type", type=str.lower, default="figure_8", 
        choices=[t.value for t in TrajectoryType]
    )
    
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument(
        "--skip-stage1", action="store_true", 
        help="Skip LSTM pre-training"
    )
    train_group.add_argument(
        "--stage3", action="store_true", 
        help="Enable Stage 3 SAC fine-tuning"
    )
    train_group.add_argument(
        "--render", action="store_true", 
        help="Enable MuJoCo viewer during training"
    )
    
    args = parser.parse_args()
    
    # Map config ID to Enum
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2]}
    args.config = CONFIG_MAP[args.config]
    
    # Map trajectory string to Enum
    args.trajectory_type = next(
        t for t in TrajectoryType 
        if t.value.lower() == args.trajectory_type.lower()
    )
    
    return args


if __name__ == "__main__":
    train_agent(parse_arguments())