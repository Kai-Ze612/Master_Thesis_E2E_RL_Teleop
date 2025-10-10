# Python imports
import os
import sys
import argparse
from datetime import datetime
import torch.nn as nn
import logging
import numpy as np

# Stable Baselines3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

# Custom imports
from training_env import TeleoperationEnvWithDelay
from custom_policy import create_predictor_policy  # ← NEW: Import custom policy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, eval_freq: int, patience: int = 10, min_improvement: float = 1.0, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.eval_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        
        self.eval_count += 1
        current_reward = self.logger.name_to_value.get('eval/mean_reward', None)
        
        if current_reward is None:
            return True
        
        improvement = current_reward - self.best_mean_reward
        
        if improvement > self.min_improvement:
            print(f"\n{'='*70}\nNEW BEST: {self.best_mean_reward:.1f} → {current_reward:.1f} (+{improvement:.1f})\n{'='*70}\n")
            self.best_mean_reward = current_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            print(f"\nNo improvement: {self.no_improvement_count}/{self.patience} (current: {current_reward:.1f}, best: {self.best_mean_reward:.1f})")
            
            if self.no_improvement_count >= self.patience:
                print(f"\n{'='*70}\nEARLY STOPPING: No improvement for {self.patience} evaluations\nBest: {self.best_mean_reward:.1f} at {self.num_timesteps} steps\n{'='*70}\n")
                return False
        
        return True

class TrackingErrorLoggingCallback(BaseCallback):
    """Log tracking errors to tensorboard."""
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.tracking_errors = []
        
    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'real_time_cartesian_error' in info:
                self.tracking_errors.append(info['real_time_cartesian_error'])
                
                if len(self.tracking_errors) % 100 == 0:
                    self.logger.record('custom/tracking_error_mean', np.mean(self.tracking_errors[-100:]))
                    if 'mean_correction_percentage' in info:
                        self.logger.record('custom/correction_pct', info['mean_correction_percentage'])
        return True

def train_agent(args):
    """Train SAC agent with learned NN predictor."""
    
    # Create run name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"Config{args.config}_NNPredictor_{timestamp}"
    
    # Setup directories
    output_dir = os.path.join(args.output_path, run_name)
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Print training configuration
    print(f"\n{'='*70}")
    print(f"TRAINING WITH LEARNED NN PREDICTOR")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Prediction: Learned Neural Network (not manual interpolation)")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seed: {args.seed if args.seed else 'Random'}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Network: {args.net_arch}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Environment configuration
    env_kwargs = {
        'model_path': args.model_path,
        'experiment_config': args.config,
        'control_freq': args.freq,
        'max_episode_steps': args.max_steps,
        'max_cartesian_error': args.max_cartesian_error,
        'use_interpolation': False,  # Not used anymore
    }

    # Create environments
    env = make_vec_env(lambda: Monitor(TeleoperationEnvWithDelay(**env_kwargs)), n_envs=1)
    eval_env = Monitor(TeleoperationEnvWithDelay(**env_kwargs))

    # Setup callbacks
    callbacks = [
        EvalCallback(
            eval_env, 
            best_model_save_path=model_dir, 
            log_path=log_dir, 
            eval_freq=args.eval_freq, 
            n_eval_episodes=args.eval_episodes, 
            deterministic=False, 
            verbose=1
        ),
        CheckpointCallback(
            save_freq=args.save_freq, 
            save_path=model_dir, 
            name_prefix="sac_nnpredictor"
        ),
        TrackingErrorLoggingCallback(verbose=1)
    ]
    
    if args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                args.eval_freq, 
                args.patience, 
                args.min_improvement, 
                verbose=1
            )
        )

    # ═══════════════════════════════════════════════════════════
    # CREATE CUSTOM POLICY WITH NN PREDICTOR
    # ═══════════════════════════════════════════════════════════
    policy_kwargs = create_predictor_policy()
    
    print(f"\n{'='*70}")
    print(f"CREATING SAC WITH CUSTOM POLICY")
    print(f"{'='*70}")
    print(f"Policy: Custom (Predictor + Controller)")
    print(f"Features dim: 256")
    print(f"Actor/Critic nets: [256, 256]")
    print(f"{'='*70}\n")

    # Create SAC model with custom policy
    model = SAC(
        'MlpPolicy', 
        env,
        learning_rate=args.learning_rate, 
        buffer_size=args.buffer_size, 
        batch_size=args.batch_size,
        ent_coef=args.ent_coef, 
        gamma=args.gamma, 
        tau=args.tau,
        learning_starts=args.learning_starts, 
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps, 
        target_update_interval=args.target_update_interval,
        seed=args.seed, 
        verbose=1, 
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs  # ← CUSTOM POLICY HERE!
    )

    # Train model
    try:
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}\n")
        
        model.learn(
            total_timesteps=args.timesteps, 
            callback=callbacks, 
            progress_bar=True, 
            tb_log_name="SAC_NNPredictor"
        )
        
        # Save final model
        final_model_path = os.path.join(model_dir, "final_model.zip")
        model.save(final_model_path)
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Final model saved to: {final_model_path}")
        print(f"Best model saved to: {os.path.join(model_dir, 'best_model.zip')}")
        print(f"Tensorboard logs: {log_dir}")
        print(f"{'='*70}\n")
            
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR DURING TRAINING")
        print(f"{'='*70}")
        print(f"Error: {e}")
        
        # Save error model for debugging
        error_model_path = os.path.join(model_dir, "error_model.zip")
        model.save(error_model_path)
        print(f"Error model saved to: {error_model_path}")
        print(f"{'='*70}\n")
        raise
        
    finally:
        env.close()
        eval_env.close()
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Train inverse dynamics + RL controller")
    parser.add_argument("--model_path", type=str, default="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml")
    parser.add_argument("--config", type=int, default=4, choices=[1,2,3,4])
    parser.add_argument("--output_path", type=str, default="./rl_training_output")
    parser.add_argument("--freq", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_cartesian_error", type=float, default=0.3)
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--save_freq", type=int, default=25000)
    parser.add_argument("--eval_freq", type=int, default=2500)
    parser.add_argument("--eval_episodes", type=int, default=30)
    
    # Interpolation arguments
    parser.add_argument("--use_interpolation", action="store_true",
                       help="Enable linear interpolation for trajectory prediction")
    parser.add_argument("--no_interpolation", dest="use_interpolation", action="store_false",
                       help="Disable interpolation (baseline)")
    parser.set_defaults(use_interpolation=True)
    
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_improvement", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--ent_coef", type=str, default="auto")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning_starts", type=int, default=20000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--net_arch", type=int, nargs='+', default=[1024, 512, 256])
    parser.add_argument("--activation_fn", type=str, default="relu", choices=["relu","tanh","elu","silu"])
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    args.activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "silu": nn.SiLU}[args.activation_fn]
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)
    
    output_dir = train_agent(args)
    print(f"\nTraining complete! Results: {output_dir}")

if __name__ == "__main__":
    main()