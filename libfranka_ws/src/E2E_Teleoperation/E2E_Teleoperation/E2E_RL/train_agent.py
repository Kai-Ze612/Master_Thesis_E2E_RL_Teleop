"""
The main training script for E2E Reinforcement Learning (LSTM + SAC) agent
With Teacher Warmup for accelerated learning

Usage:
    python train_agent.py --config 3 --render human
    python train_agent.py --config 3 --teacher-warmup-steps 20000
"""
import os
import sys
from datetime import datetime
import logging
import torch
import numpy as np
import argparse
import multiprocessing

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnvWithDelay
from E2E_Teleoperation.E2E_RL.local_robot_simulator import TrajectoryType
from E2E_Teleoperation.E2E_RL.sac_training_algorithm import SACTrainer, SACConfig
from E2E_Teleoperation.E2E_RL.sac_policy_network import SharedLSTMEncoder, JointActor, JointCritic
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig
import E2E_Teleoperation.config.robot_config as cfg


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
    run_name = f"E2E_SAC_TeacherWarmup_{args.config.name}_{args.trajectory_type.value}_{timestamp}"
    output_dir = os.path.join(cfg.CHECKPOINT_DIR_RL, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("E2E SAC Training with TEACHER WARMUP")
    logger.info("=" * 70)
    
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
                render_mode=args.render
            )
        env = make_vec_env(make_env, n_envs=cfg.NUM_ENVIRONMENTS, seed=args.seed, vec_env_cls=SubprocVecEnv)
    except Exception as e:
        logger.error(f"Failed to create env: {e}", exc_info=True)
        if env: env.close()
        sys.exit(1)
   
    logger.info("Creating policy network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        shared_encoder = SharedLSTMEncoder().to(device)
        policy_network = JointActor(shared_encoder=shared_encoder, action_dim=cfg.N_JOINTS).to(device)
    except Exception as e:
        logger.error(f"Failed to create policy network: {e}", exc_info=True)
        env.close()
        sys.exit(1)

    # [FIX] Enforce Alpha Clamping in Config
    sac_config = SACConfig(
        actor_lr=cfg.ACTOR_LR if hasattr(cfg, 'ACTOR_LR') else 3e-4,
        critic_lr=cfg.CRITIC_LR if hasattr(cfg, 'CRITIC_LR') else 3e-4,
        alpha_lr=cfg.ALPHA_LR if hasattr(cfg, 'ALPHA_LR') else 3e-4,
        encoder_lr=cfg.ENCODER_LR if hasattr(cfg, 'ENCODER_LR') else 1e-3,
        gamma=cfg.GAMMA if hasattr(cfg, 'GAMMA') else 0.99,
        tau=cfg.TAU if hasattr(cfg, 'TAU') else 0.005,
        initial_alpha=0.05,
        alpha_min=cfg.ALPHA_MIN if hasattr(cfg, 'ALPHA_MIN') else 0.01,
        alpha_max=0.2, # [FIX] Explicitly set max alpha
        target_entropy_scale=0.5,
        fixed_alpha=False,
        encoder_warmup_steps=cfg.SAC_START_STEPS,
        teacher_warmup_steps=args.teacher_warmup_steps,
        buffer_size=cfg.BUFFER_SIZE if hasattr(cfg, 'BUFFER_SIZE') else 1000000,
        batch_size=cfg.BATCH_SIZE if hasattr(cfg, 'BATCH_SIZE') else 256,
        policy_delay=2,
        validation_freq=cfg.VAL_FREQ if hasattr(cfg, 'VAL_FREQ') else 10000,
        checkpoint_freq=cfg.CHECKPOINT_FREQ if hasattr(cfg, 'CHECKPOINT_FREQ') else 50000,
        log_freq=cfg.LOG_FREQ if hasattr(cfg, 'LOG_FREQ') else 1000,
    )

    logger.info("Initializing SACTrainer with Teacher Warmup...")
    trainer = None
    try:
        trainer = SACTrainer(env=env, policy_network=policy_network, config=sac_config, output_dir=output_dir, device=device)
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}", exc_info=True)
        env.close()
        sys.exit(1)
   
    try:
        trainer.train(total_timesteps=args.timesteps)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        trainer.save_checkpoint("interrupted_policy.pth")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        trainer.save_checkpoint("crash_policy.pth")
    finally:
        if env: env.close()
        logger.info("Training finished.")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E SAC Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument("--config", type=str, default="3", choices=['1', '2', '3'], help="Delay config")
    exp_group.add_argument("--trajectory-type", type=str.lower, default="figure_8", choices=[t.value for t in TrajectoryType])
    exp_group.add_argument("--randomize-trajectory", action="store_true")
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--timesteps", type=int, default=cfg.SAC_TOTAL_TIMESTEPS)
    train_group.add_argument("--teacher-warmup-steps", type=int, default=30000) # [REC] 30k
    train_group.add_argument("--seed", type=int, default=None)
    train_group.add_argument("--render", type=str.lower, default=None, choices=['human', 'rgb_array', 'none'])
    args = parser.parse_args()
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2]}
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    return args

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    train_agent(parse_arguments())