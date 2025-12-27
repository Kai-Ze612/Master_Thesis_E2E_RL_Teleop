"""
E2E_Teleoperation/E2E_RL/train_agent.py
"""

import argparse
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import sys

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
from E2E_Teleoperation.E2E_RL.unified_trainer import UnifiedTrainer
from E2E_Teleoperation.E2E_RL.local_robot_simulator import TrajectoryType
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig

def load_checkpoint(trainer, load_dir, filename, model_attr):
    """
    Helper to load weights from a previous run into the trainer.
    model_attr: 'encoder' or 'actor' (attribute name in UnifiedTrainer)
    """
    if not load_dir:
        print(f"[ERROR] You are trying to skip stages but didn't provide --load-dir!")
        sys.exit(1)
        
    path = Path(load_dir) / filename
    if not path.exists():
        print(f"[ERROR] Checkpoint not found at: {path}")
        sys.exit(1)
        
    print(f">> Loading {model_attr.upper()} from: {filename}...")
    
    # Access the model inside trainer (trainer.encoder or trainer.actor)
    model = getattr(trainer, model_attr)
    model.load_state_dict(torch.load(path, map_location=trainer.device))

def train_agent(args):
    # 1. Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"E2E_RL_{args.config.name}_{args.trajectory_type.name}_{timestamp}"
    
    output_dir = cfg.ROBOT.CHECKPOINT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"==================================================")
    print(f"STARTING TRAINING: {run_name}")
    print(f"Delay Config:   {args.config.name}")
    print(f"Trajectory:     {args.trajectory_type.name}")
    print(f"Start Stage:    {args.start_stage}")
    if args.load_dir:
        print(f"Loading from:   {args.load_dir}")
    print(f"==================================================")
    
    # 2. Environment
    env = TeleoperationEnv(
        delay_config=args.config,
        trajectory_type=args.trajectory_type,
        seed=args.seed,
        render_mode="human" if args.render else None
    )
    
    # 3. Trainer (Uses UnifiedTrainer logic)
    trainer = UnifiedTrainer(env, str(output_dir))
    
    # 4. Run Stages with Skip Logic
    
    # --- STAGE 1: Encoder ---
    if args.start_stage == 1:
        trainer.train_stage1()
    else:
        # If we skip Stage 1, we MUST load the pretrained Encoder
        load_checkpoint(trainer, args.load_dir, "stage1_final.pth", "encoder")

    # --- STAGE 2: BC ---
    if args.start_stage <= 2:
        trainer.train_stage2_bc()
    else:
        # If we skip Stage 2, we MUST load the pretrained Actor (BC Policy)
        load_checkpoint(trainer, args.load_dir, "stage2_final.pth", "actor")

    # --- STAGE 3: SAC ---
    # Always runs (unless you added a stage 4, which doesn't exist yet)
    trainer.train_stage3_sac()
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Config arguments
    parser.add_argument(
        "--config", type=str, default="3", choices=['1', '2', '3'], 
        help="Delay Profile: 1=Low, 2=High, 3=Var"
    )
    parser.add_argument(
        "--trajectory-type", type=str, default="FIGURE_8", 
        choices=[t.name for t in TrajectoryType]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    
    # --- NEW ARGUMENTS FOR SKIPPING STAGES ---
    parser.add_argument(
        "--start-stage", type=int, default=1, choices=[1, 2, 3],
        help="1=Full Training, 2=Skip Encoder, 3=Skip BC (Jump to SAC)"
    )
    parser.add_argument(
        "--load-dir", type=str, default=None,
        help="Directory containing stage1_final.pth and stage2_final.pth (Required if start-stage > 1)"
    )
    
    args = parser.parse_args()
    
    # Map arguments
    CONFIG_MAP = {
        '1': ExperimentConfig.LOW_DELAY,
        '2': ExperimentConfig.HIGH_DELAY,
        '3': ExperimentConfig.HIGH_VARIANCE
    }
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = TrajectoryType[args.trajectory_type.upper()]
    
    train_agent(args)