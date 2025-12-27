"""
The starting point for training.

Features:
- Supports both single and vectorized environments.

argparse Arguments:
--num-envs: Number of parallel environments to run (default: 1)
--config: Delay configuration to use (1, 2, or 3) (default: 3)
--trajectory-type: Type of trajectory to train on (default: FIGURE_8)
--seed: Random seed for reproducibility (default: 42)
--render: Whether to render the environment during training (default: False)
--start-stage: Training stage to start from (1, 2, or 3) (default: 1), if 3, skips to SAC training
--load-dir: Directory to load pre-trained models from (default: None)
"""

import argparse
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import sys
import gymnasium as gym

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.unified_trainer import UnifiedTrainer
from E2E_Teleoperation.E2E_RL.local_robot_simulator import TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

def load_checkpoint(trainer, load_dir, filename, model_attr):
    """
    Helper to load weights from a previous run into the trainer.
    """
    if not load_dir:
        print(f"The pretrained model directory is not specified. Cannot load {model_attr}.")
        sys.exit(1)
        
    path = Path(load_dir) / filename
    if not path.exists():
        print(f"The specified checkpoint file does not exist: {path}")
        sys.exit(1)
        
    print(f"Loaded pretrained {model_attr} from: {path}")
    
    model = getattr(trainer, model_attr)
    model.load_state_dict(torch.load(path, map_location=trainer.device))

def make_env(args, rank):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = TeleoperationEnv(
            delay_config=args.config,
            trajectory_type=args.trajectory_type,
            randomize_trajectory=args.randomize_trajectory,
            seed=args.seed + rank,  
            render_mode="human" if args.render else None
        )
        return env
    return _init

def train_agent(args):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"E2E_RL_{args.config.name}_{args.trajectory_type.name}_{timestamp}"
    
    output_dir = cfg.ROBOT.CHECKPOINT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"==================================================")
    print(f"STARTING TRAINING: {run_name}")
    print(f"Environments:   {args.num_envs}")
    print(f"Trajectory:     {args.trajectory_type.name}")
    print(f"Randomized:     {args.randomize_trajectory}")
    print(f"Start Stage:    {args.start_stage}")
    print(f"Delay Config:   {args.config.name}")
    print(f"==================================================")
    
    if args.num_envs == 1:
        # Standard Single Environment
        env = TeleoperationEnv(
            delay_config=args.config,
            trajectory_type=args.trajectory_type,
            randomize_trajectory=args.randomize_trajectory,
            seed=args.seed,
            render_mode="human" if args.render else None
        )
    else:
        # Vectorized Environment
        env = gym.vector.AsyncVectorEnv(
            [make_env(args, i) for i in range(args.num_envs)]
        )

    trainer = UnifiedTrainer(env, str(output_dir), is_vector_env=(args.num_envs > 1))
    
    # --- STAGE 1: Encoder ---
    if args.start_stage == 1:
        trainer.train_stage1()
    else:
        load_checkpoint(trainer, args.load_dir, "stage1_best.pth", "encoder")

    # --- STAGE 2: BC ---
    if args.start_stage <= 2:
        trainer.train_stage2_bc()
    else:
        load_checkpoint(trainer, args.load_dir, "stage2_best.pth", "actor")

    # --- STAGE 3: SAC ---
    trainer.train_stage3_sac()
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Experiment Config arguments
    parser.add_argument("--config", type=str, default="3", choices=['1', '2', '3'])
    parser.add_argument("--trajectory-type", type=str, default="FIGURE_8", choices=[t.name for t in TrajectoryType])
    parser.add_argument("--randomize-trajectory", action="store_true", 
                        help="Randomize trajectory parameters (scale, speed) per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    
    # Checkpoint / Staging Arguments
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--load-dir", type=str, default=None,
                        help="Path to directory containing .pth files for skipping stages")

    # Vector Env Argument
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments to run")
    
    args = parser.parse_args()
    
    # Map Config
    CONFIG_MAP = {
        '1': ExperimentConfig.LOW_DELAY,
        '2': ExperimentConfig.HIGH_DELAY,
        '3': ExperimentConfig.HIGH_VARIANCE
    }
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = TrajectoryType[args.trajectory_type.upper()]
    
    train_agent(args)