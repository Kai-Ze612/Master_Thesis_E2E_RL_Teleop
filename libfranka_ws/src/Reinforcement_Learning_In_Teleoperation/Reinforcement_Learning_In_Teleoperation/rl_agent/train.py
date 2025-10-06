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

from training_env import TeleoperationEnvWithDelay

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStoppingCallback(BaseCallback):
    """Stop training if no improvement for N evaluations."""
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


def test_baseline_vs_rl(env, model, n_episodes=10):
    """Compare baseline (action=0) vs RL policy."""
    print(f"\n{'='*70}\nBASELINE EVALUATION (No RL)\n{'='*70}")
    
    baseline_errors, baseline_rewards = [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, step_count = False, 0
        ep_errors, ep_rewards = [], []
        
        while not done and step_count < env.max_episode_steps:
            obs, reward, terminated, truncated, info = env.step(np.zeros(env.n_joints))
            done = terminated or truncated
            ep_errors.append(info['real_time_cartesian_error'])
            ep_rewards.append(reward)
            step_count += 1
        
        baseline_errors.append(np.mean(ep_errors))
        baseline_rewards.append(np.sum(ep_rewards))
        print(f"  Ep {ep+1}: Error={np.mean(ep_errors):.4f}m, Reward={np.sum(ep_rewards):.0f}, Steps={step_count}")
    
    print(f"\nBaseline: Error={np.mean(baseline_errors):.4f}±{np.std(baseline_errors):.4f}m, Reward={np.mean(baseline_rewards):.0f}±{np.std(baseline_rewards):.0f}")
    
    print(f"\n{'='*70}\nRL POLICY EVALUATION\n{'='*70}")
    
    rl_errors, rl_rewards, rl_actions = [], [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, step_count = False, 0
        ep_errors, ep_rewards, ep_actions = [], [], []
        
        while not done and step_count < env.max_episode_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_errors.append(info['real_time_cartesian_error'])
            ep_rewards.append(reward)
            ep_actions.append(np.abs(action))
            step_count += 1
        
        rl_errors.append(np.mean(ep_errors))
        rl_rewards.append(np.sum(ep_rewards))
        rl_actions.extend(ep_actions)
        print(f"  Ep {ep+1}: Error={np.mean(ep_errors):.4f}m, Reward={np.sum(ep_rewards):.0f}, Steps={step_count}")
    
    print(f"\nRL Policy: Error={np.mean(rl_errors):.4f}±{np.std(rl_errors):.4f}m, Reward={np.mean(rl_rewards):.0f}±{np.std(rl_rewards):.0f}, |Action|={np.mean(rl_actions):.3f}")
    
    error_improvement = (np.mean(baseline_errors) - np.mean(rl_errors)) / np.mean(baseline_errors) * 100
    
    print(f"\n{'='*70}\nCOMPARISON\n{'='*70}")
    print(f"Error improvement: {error_improvement:+.1f}%")
    
    if error_improvement > 10:
        print("VERDICT: ✓ RL helps significantly")
    elif error_improvement > 0:
        print("VERDICT: ~ RL helps slightly")
    else:
        print("VERDICT: ✗ RL not helping")
    print(f"{'='*70}\n")
    
    return {
        'baseline_error': np.mean(baseline_errors),
        'rl_error': np.mean(rl_errors),
        'error_improvement': error_improvement,
        'mean_action': np.mean(rl_actions)
    }


## Training starts
def train_agent(args):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"SAC_InvDyn_Config{args.config}_{timestamp}"
    output_dir = os.path.join(args.output_path, run_name)
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"{'='*70}\nINVERSE DYNAMICS + RL TRAINING\nConfig {args.config}, {args.freq}Hz, {args.max_steps} steps/ep\nEarly stopping: {args.early_stopping} (patience={args.patience if args.early_stopping else 'N/A'})\n{'='*70}")

    env_kwargs = {
        'model_path': args.model_path,
        'experiment_config': args.config,
        'control_freq': args.freq,
        'max_episode_steps': args.max_steps,
        'max_cartesian_error': args.max_cartesian_error,
    }

    env = make_vec_env(lambda: Monitor(TeleoperationEnvWithDelay(**env_kwargs)), n_envs=1)
    eval_env = Monitor(TeleoperationEnvWithDelay(**env_kwargs))

    callbacks = [
        EvalCallback(eval_env, best_model_save_path=model_dir, log_path=log_dir, 
                    eval_freq=args.eval_freq, n_eval_episodes=args.eval_episodes, 
                    deterministic=False, verbose=1),
        CheckpointCallback(save_freq=args.save_freq, save_path=model_dir, name_prefix="sac_invdyn"),
        TrackingErrorLoggingCallback(verbose=1)
    ]
    
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(args.eval_freq, args.patience, args.min_improvement, verbose=1))

    model = SAC(
        'MlpPolicy', env,
        learning_rate=args.learning_rate, buffer_size=args.buffer_size, batch_size=args.batch_size,
        ent_coef=args.ent_coef, gamma=args.gamma, tau=args.tau,
        learning_starts=args.learning_starts, train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps, target_update_interval=args.target_update_interval,
        seed=args.seed, verbose=1, tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=args.net_arch, activation_fn=args.activation_fn, optimizer_kwargs=dict(eps=1e-5))
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True, tb_log_name="SAC_InvDyn")
        model.save(os.path.join(model_dir, "final_model.zip"))
        
        print(f"\n{'='*70}\nPOST-TRAINING EVALUATION\n{'='*70}")
        results = test_baseline_vs_rl(eval_env, model, n_episodes=20)
        
        import json
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        print(f"\nError: {e}")
        model.save(os.path.join(model_dir, "error_model.zip"))
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
    parser.add_argument("--max_cartesian_error", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--save_freq", type=int, default=25000)
    parser.add_argument("--eval_freq", type=int, default=2500)
    parser.add_argument("--eval_episodes", type=int, default=10)
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
    parser.add_argument("--net_arch", type=int, nargs='+', default=[512, 256])
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