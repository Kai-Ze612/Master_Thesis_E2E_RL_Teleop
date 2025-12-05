"""
Training script for Model-Based RL using SBSP (PMDC) Wrapper.
Includes Validation Wrapper Injection for correct evaluation.
"""

import os
import sys
import argparse
import numpy as np
import torch
import logging
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv

# Import User's Environment and Config
# Ensure you run this from the project root!
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_training_algorithm import SACTrainer
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import CHECKPOINT_DIR_RL

# Import Adapted Wrapper
from sbsp_wrapper import SBSP_Trajectory_Wrapper

def setup_logging(output_dir: str):
    log_file = os.path.join(output_dir, "sbsp_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def make_sbsp_env(args):
    """
    Factory function to create the Env wrapped in SBSP.
    """
    # 1. Create Base Env
    # Note: We pass None to lstm_model_path because we are replacing it with SBSP
    env = TeleoperationEnvWithDelay(
        delay_config=args.config,
        trajectory_type=args.trajectory_type,
        randomize_trajectory=args.randomize_trajectory,
        render_mode=args.render,
        lstm_model_path=None 
    )
    
    # 2. Wrap with SBSP (PMDC)
    env = SBSP_Trajectory_Wrapper(env, n_models=5)
    return env

def train_sbsp_agent(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"SBSP_SAC_{args.config.name}_{args.trajectory_type.value}_{timestamp}"
    output_dir = os.path.join(CHECKPOINT_DIR_RL, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("Initializing SBSP Training...")
    
    # 1. Create Vectorized Environment
    env = DummyVecEnv([lambda: make_sbsp_env(args)])

    logger.info("Environment and SBSP Wrapper Initialized.")

    # 2. Initialize SACTrainer
    # This creates the trainer AND its own internal 'self.val_env' (unwrapped)
    try:
        trainer = SACTrainer(env=env, val_delay_config=args.config)
        trainer.checkpoint_dir = output_dir
        
        # --- FIX: Inject SBSP into Validation ---
        logger.info("Injecting SBSP Wrapper into Validation Environment...")
        
        # A. Get the TRAINED wrapper from the training env
        # DummyVecEnv stores environments in the .envs list
        logger.info("Injecting SBSP Wrapper into Validation Environment...")
        train_wrapper = env.envs[0] 
        val_wrapped = SBSP_Trajectory_Wrapper(trainer.val_env, n_models=5)
        val_wrapped.dc_models = train_wrapper.dc_models # <--- Sharing Weights
        trainer.val_env = val_wrapped
        logger.info("Validation Environment successfully patched with shared SBSP models.")
        # ----------------------------------------

        logger.info("Starting Training Loop...")
        trainer.train(total_timesteps=args.timesteps)
        
    except KeyboardInterrupt:
        logger.warning("Training Interrupted by User (Ctrl+C).")
        if 'trainer' in locals():
            trainer.save_checkpoint("interrupted_sbsp_policy.pth")
    except Exception as e:
        logger.error(f"Training Failed: {e}", exc_info=True)
        if 'trainer' in locals():
            trainer.save_checkpoint("crash_sbsp_policy.pth")
    finally:
        env.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="2", choices=['1', '2', '3', '4'])
    parser.add_argument("--trajectory-type", type=str.lower, default="figure_8", choices=[t.value for t in TrajectoryType])
    parser.add_argument("--randomize-trajectory", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--render", type=str.lower, default=None)
    
    args = parser.parse_args()
    
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2], '4': config_options[3]}
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    train_sbsp_agent(args)