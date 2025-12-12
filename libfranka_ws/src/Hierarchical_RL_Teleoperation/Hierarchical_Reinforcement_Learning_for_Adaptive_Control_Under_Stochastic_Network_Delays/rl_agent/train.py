import os
import sys
import argparse
from datetime import datetime
import torch.nn as nn
import logging
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from training_env import TeleoperationEnvWithDelay
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_agent(args):
    """Sets up the delay-aware environment, callbacks, and agent, then starts training."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"SAC_DelayConfig{args.config}_Freq{args.freq}_{args.prediction_method}_{timestamp}"
    output_dir = os.path.join(args.output_path, run_name)
    log_dir, model_dir = os.path.join(output_dir, "logs"), os.path.join(output_dir, "models")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("-" * 70)
    print(f"TRAINING CONFIGURATION")
    print(f"Delay Config: {args.config}")
    print(f"Prediction Method: {args.prediction_method}")
    print(f"Control Frequency: {args.freq} Hz")
    print(f"Max Episode Steps: {args.max_steps}")
    print(f"Output Directory: {output_dir}")
    print("-" * 70)

    # Clean env_kwargs without any LSTM dependencies
    env_kwargs = {
        'model_path': args.model_path,
        'experiment_config': args.config,
        'control_freq': args.freq,
        'max_episode_steps': args.max_steps,
        'max_cartesian_error': 0.5,
        'prediction_method': args.prediction_method
    }

    # Create environments
    env = make_vec_env(lambda: Monitor(TeleoperationEnvWithDelay(**env_kwargs)), n_envs=1)
    eval_env = Monitor(TeleoperationEnvWithDelay(**env_kwargs))

    # Standard callbacks without early stopping
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq, 
        save_path=model_dir, 
        name_prefix="sac_model"
    )

    # Enhanced SAC model
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
        policy_kwargs=dict(
            net_arch=args.net_arch,
            activation_fn=args.activation_fn,
            optimizer_kwargs=dict(eps=1e-5)
        )
    )

    print(f"Starting training with {args.timesteps:,} timesteps...")
    print("No early stopping - training will run for full duration")
    print("="*70)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
            tb_log_name="SAC"
        )
        final_model_path = os.path.join(model_dir, "final_model.zip")
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
            
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        error_model_path = os.path.join(model_dir, "error_model.zip")
        model.save(error_model_path)
        print(f"Backup model saved to: {error_model_path}")
        raise
    finally:
        env.close()
        eval_env.close()
    
    print("--- Training Complete ---")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Train a delay-aware SAC agent with physics-based prediction.")
    default_model_path = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml"
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to the MuJoCo XML model file.")
    parser.add_argument("--config", type=int, default=1, choices=[1, 2, 3], help="Delay experiment config 1 to 3")
    parser.add_argument("--prediction_method", type=str, default="linear_extrapolation", 
                       choices=["linear_extrapolation", "velocity_based", "delayed_obs"], 
                       help="Physics-based prediction method")
    parser.add_argument("--output_path", type=str, default="./rl_training_output", help="Directory to save logs and models.")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total number of training timesteps.")
    parser.add_argument("--freq", type=int, default=500, help="Control frequency in Hz.")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--save_freq", type=int, default=25000, help="Frequency to save model checkpoints.")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Frequency to run model evaluations.")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of episodes for each evaluation.")
    
    # Enhanced hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--buffer_size", type=int, default=500000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--ent_coef", type=str, default="auto", help="Entropy coefficient ('auto' for adaptive or float value)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient.")
    parser.add_argument("--learning_starts", type=int, default=10000, help="Steps before training starts")
    parser.add_argument("--train_freq", type=int, default=1, help="Training frequency in steps")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Gradient steps per training update")
    parser.add_argument("--target_update_interval", type=int, default=1, help="Target network update interval.")
    parser.add_argument("--net_arch", type=int, nargs='+', default=[512, 512, 256], help="Network architecture")
    parser.add_argument("--activation_fn", type=str, default="elu", choices=["relu", "tanh", "elu", "silu"], help="Activation function for neural networks.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training.")

    args = parser.parse_args()

    # Convert activation function string to actual function
    if args.activation_fn == "relu":
        args.activation_fn = nn.ReLU
    elif args.activation_fn == "tanh":
        args.activation_fn = nn.Tanh
    elif args.activation_fn == "elu":
        args.activation_fn = nn.ELU
    elif args.activation_fn == "silu":
        args.activation_fn = nn.SiLU

    if not os.path.exists(args.model_path):
        print(f"Error: MuJoCo model path does not exist: {args.model_path}")
        sys.exit(1)

    # Get delay config name for display
    delay_config_names = {
        1: "Low Delay",
        2: "Medium Delay", 
        3: "High Delay, High Variance"
    }

    print("=" * 70)
    print("RL TRAINING WITH PHYSICS-BASED DELAY COMPENSATION")
    print("=" * 70)
    print(f"Config: {args.config} - {delay_config_names.get(args.config, 'Unknown')}")
    print(f"Prediction Method: {args.prediction_method}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Network Architecture: {args.net_arch}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Buffer Size: {args.buffer_size:,}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Activation Function: {args.activation_fn.__name__}")
    print("LSTM: DISABLED (physics-based prediction only)")
    print("=" * 70)

    # Run training
    output_dir = train_agent(args)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()