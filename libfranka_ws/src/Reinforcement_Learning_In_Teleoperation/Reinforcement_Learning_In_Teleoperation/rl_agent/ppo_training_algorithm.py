"""
Custom recurrent PPO policy for end to end training.

This script defines the training loop and optimization process for the recurrent PPO agent.

Combines:
- Supervised loss for state prediction (RNN)
- PPO loss for RL compensation (Actor-Critic)
- Handles RNN hidden state management during training.
- Dense reward computed in environment (prediction + tracking accuracy)

Training pipeline:
    1. Collect rollout: Policy predicts state → takes action → environment calculates reward
    2. Compute GAE: Calculate advantages and returns
    3. Update policy: PPO loss + Supervised prediction loss
    4. Check early stopping: Save best model if improvement
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional
import time
import json
from torch.utils.tensorboard import SummaryWriter

from Reinforcement_Learning_In_Teleoperation.rl_agent.ppo_policy_network import RecurrentPPOPolicy, HiddenStateType
from Reinforcement_Learning_In_Teleoperation.utils.rollout_buffer import RolloutBuffer
from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS, RNN_SEQUENCE_LENGTH,
    PPO_LEARNING_RATE, PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_EPSILON,
    PPO_ENTROPY_COEF, PPO_VALUE_COEF, PPO_MAX_GRAD_NORM,
    PREDICTION_LOSS_WEIGHT, PPO_LOSS_WEIGHT,
    PPO_ROLLOUT_STEPS, PPO_NUM_EPOCHS, PPO_BATCH_SIZE,
    LOG_FREQ, SAVE_FREQ, CHECKPOINT_DIR,
    ENABLE_EARLY_STOPPING, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_CHECK_FREQ
)

logger = logging.getLogger(__name__)


class RecurrentPPOTrainer:

    def __init__(self, env, device: torch.device):
        
        self.env = env
        self.device = device

        # Initialize policy
        self.policy = RecurrentPPOPolicy().to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=PPO_LEARNING_RATE,
            eps=1e-5  # Adam epsilon for numerical stability
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=PPO_ROLLOUT_STEPS,
            device=self.device
        )

        # Training state
        self.total_timesteps = 0
        self.num_updates = 0
        self.checkpoint_dir = CHECKPOINT_DIR  # Default from config, can be overridden

        # Visualization setup
        self.tb_writer: Optional[SummaryWriter] = None
        
        # Metrics storage for JSON logging
        self.metrics_history = {
            'steps': [],
            'timestamps': [],
            'rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'prediction_losses': [],
            'entropies': [],
            'approx_kls': [],
            'total_losses': [],
        }
        
        self.training_start_time = None

        # Early Stopping Attributes
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_policy_earlystop.pth")

    def _init_tensorboard(self, log_dir: str):
        """Initialize TensorBoard writer."""
        tb_dir = os.path.join(log_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=120)
        logger.info(f"  Launch with: tensorboard --logdir {tb_dir}")
        logger.info(f"  Then open: http://localhost:6006\n")

    def _log_metrics(self, metrics: Dict[str, float], avg_reward: float):
        """
        Log metrics to both TensorBoard and JSON storage.
        
        Args:
            metrics: Dictionary of training metrics
            avg_reward: Average episode reward
        """
        step = self.num_updates
        
        # Store for JSON export
        elapsed = (datetime.now() - self.training_start_time).total_seconds()
        self.metrics_history['steps'].append(step)
        self.metrics_history['timestamps'].append(elapsed)
        self.metrics_history['rewards'].append(avg_reward)
        self.metrics_history['actor_losses'].append(metrics.get('actor_loss', 0.0))
        self.metrics_history['critic_losses'].append(metrics.get('critic_loss', 0.0))
        self.metrics_history['prediction_losses'].append(metrics.get('prediction_loss', 0.0))
        self.metrics_history['entropies'].append(metrics.get('entropy', 0.0))
        self.metrics_history['approx_kls'].append(metrics.get('approx_kl', 0.0))
        self.metrics_history['total_losses'].append(metrics.get('total_loss', 0.0))
        
        # Log to TensorBoard if available
        if self.tb_writer:
            # Main reward
            self.tb_writer.add_scalar('train/avg_reward', avg_reward, step)
            
            # Loss components (grouped)
            self.tb_writer.add_scalars('losses', {
                'actor': metrics.get('actor_loss', 0.0),
                'critic': metrics.get('critic_loss', 0.0),
                'prediction': metrics.get('prediction_loss', 0.0),
                'total': metrics.get('total_loss', 0.0)
            }, step)
            
            # Policy statistics
            self.tb_writer.add_scalar('train/entropy', metrics.get('entropy', 0.0), step)
            self.tb_writer.add_scalar('train/approx_kl', metrics.get('approx_kl', 0.0), step)
            
            # Flush periodically
            if step % (LOG_FREQ * 2) == 0:
                self.tb_writer.flush()

    def _save_metrics_json(self):
        """Save all metrics to JSON file."""
        metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.json")
        
        output = {
            'metadata': {
                'start_time': self.training_start_time.isoformat() if self.training_start_time else None,
                'end_time': datetime.now().isoformat(),
                'total_updates': len(self.metrics_history['steps']),
                'total_duration_seconds': (datetime.now() - self.training_start_time).total_seconds() if self.training_start_time else 0,
            },
            'metrics': self.metrics_history
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Training metrics saved to: {metrics_file}")

    def _save_training_summary(self):
        """Save human-readable training summary."""
        summary_file = os.path.join(self.checkpoint_dir, "training_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Training Summary\n")
            f.write("="*70 + "\n\n")
            
            if self.training_start_time:
                f.write(f"Start Time: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                duration = (datetime.now() - self.training_start_time).total_seconds()
                f.write(f"Duration: {duration / 3600:.2f} hours ({duration / 60:.1f} minutes)\n\n")
            
            f.write(f"Total Updates: {len(self.metrics_history['steps'])}\n")
            f.write(f"Total Timesteps: {self.total_timesteps:,}\n\n")
            
            if self.metrics_history['rewards']:
                rewards = self.metrics_history['rewards']
                f.write("Reward Statistics:\n")
                f.write(f"  Initial: {rewards[0]:.3f}\n")
                f.write(f"  Final: {rewards[-1]:.3f}\n")
                f.write(f"  Best: {max(rewards):.3f}\n")
                f.write(f"  Mean: {sum(rewards)/len(rewards):.3f}\n")
                f.write(f"  Std: {np.std(rewards):.3f}\n\n")
            
            if self.metrics_history['prediction_losses']:
                pred_losses = self.metrics_history['prediction_losses']
                f.write("Prediction Loss (LSTM):\n")
                f.write(f"  Initial: {pred_losses[0]:.6f}\n")
                f.write(f"  Final: {pred_losses[-1]:.6f}\n")
                f.write(f"  Best: {min(pred_losses):.6f}\n\n")
            
            if self.best_mean_reward > -np.inf:
                f.write(f"Best Reward Achieved: {self.best_mean_reward:.3f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"Training summary saved to: {summary_file}")

    def collect_rollout(self) -> float:

        self.buffer.reset()  # Clear buffer before collecting new data
        
        # Reset environment
        try:
            obs_dict = self.env.reset()
            if isinstance(obs_dict, tuple):
                initial_obs = obs_dict[0]
            else:
                initial_obs = obs_dict
                
            if not isinstance(initial_obs, np.ndarray):
                logger.warning(f"env.reset() did not return NumPy array. Type: {type(initial_obs)}")
                
        except Exception as e:
            logger.error(f"Error during environment reset: {e}", exc_info=True)
            return -np.inf

        # Initialize LSTM hidden state for the beginning of the rollout
        hidden_state: HiddenStateType = self.policy.init_hidden_state(batch_size=1, device=self.device)

        episode_rewards: List[float] = []
        current_episode_reward: float = 0.0
        steps_collected: int = 0

        while steps_collected < PPO_ROLLOUT_STEPS:
            # --- Get data from environment ---
            delayed_buffer_flat = self.env.get_delayed_target_buffer(RNN_SEQUENCE_LENGTH)
            delayed_sequence = delayed_buffer_flat.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2)

            remote_state = self.env.get_remote_state()
            true_target = self.env.get_true_current_target()

            # --- Convert to tensors ---
            delayed_sequence_t = torch.FloatTensor(delayed_sequence).unsqueeze(0).to(self.device)
            remote_state_t = torch.FloatTensor(remote_state).unsqueeze(0).to(self.device)

            # --- Get action from policy ---
            with torch.no_grad():
                action_t, log_prob_t, value_t, predicted_target_t, new_hidden_state = self.policy.get_action(
                    delayed_sequence_t,
                    remote_state_t,
                    hidden_state,
                    deterministic=False
                )

            # Convert tensors to numpy
            action_np = action_t.cpu().numpy().squeeze(axis=0)
            predicted_target_np = predicted_target_t.cpu().numpy().squeeze(axis=0)
            log_prob_val = log_prob_t.item() if log_prob_t is not None else 0.0
            value_val = value_t.item()

            # --- CRITICAL: Pass prediction to env BEFORE stepping ---
            self.env.set_predicted_target(predicted_target_np)

            # --- Step environment ---
            try:
                next_obs_tuple = self.env.step(action_np)
                
                if len(next_obs_tuple) == 5:
                    next_obs, reward, terminated, truncated, info = next_obs_tuple
                    done = terminated or truncated
                elif len(next_obs_tuple) == 4:
                    next_obs, reward, done, info = next_obs_tuple
                else:
                    logger.error(f"Unexpected env.step() return: {len(next_obs_tuple)} values")
                    # Use default values and continue
                    reward = 0.0
                    done = False
                    
            except Exception as e:
                logger.error(f"Error during environment step {steps_collected}: {e}", exc_info=True)
                # Use default values and continue instead of breaking
                reward = 0.0
                done = False

            # --- Store transition in buffer ---
            self.buffer.add(
                delayed_sequence=delayed_sequence,
                remote_state=remote_state,
                action=action_np,
                log_prob=log_prob_val,
                value=value_val,
                reward=float(reward),
                done=bool(done),
                predicted_target=predicted_target_np,
                true_target=true_target
            )

            current_episode_reward += float(reward)
            self.total_timesteps += 1
            steps_collected += 1

            # --- Update LSTM hidden state ---
            hidden_state = new_hidden_state

            # --- Handle episode end ---
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                
                reset_output = self.env.reset()
                if isinstance(reset_output, tuple):
                    obs = reset_output[0]
                else:
                    obs = reset_output
                    
                hidden_state = self.policy.init_hidden_state(batch_size=1, device=self.device)

        # --- Safety Check: Ensure we collected enough steps ---
        if steps_collected < PPO_ROLLOUT_STEPS:
            logger.error(f"Only collected {steps_collected}/{PPO_ROLLOUT_STEPS} steps. Buffer may be incomplete!")
            # Return early with error signal
            return -np.inf
        
        # --- Verify buffer has data ---
        if len(self.buffer.rewards) == 0:
            logger.error("Buffer is empty after collection! This should not happen.")
            return -np.inf

        # --- Calculate last value for GAE ---
        last_delayed_buffer = self.env.get_delayed_target_buffer(RNN_SEQUENCE_LENGTH)
        last_remote_state = self.env.get_remote_state()
        last_delayed_sequence = last_delayed_buffer.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2)
        last_delayed_sequence_t = torch.FloatTensor(last_delayed_sequence).unsqueeze(0).to(self.device)
        last_remote_state_t = torch.FloatTensor(last_remote_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, _, last_value_t, _ = self.policy.forward(
                last_delayed_sequence_t,
                last_remote_state_t,
                hidden_state
            )
        last_value = last_value_t.item()

        # --- Compute Advantages and Returns ---
        self.advantages, self.returns = self.buffer.compute_returns_and_advantages(
            last_value, PPO_GAMMA, PPO_GAE_LAMBDA
        )

        avg_episode_reward = np.mean(episode_rewards) if episode_rewards else np.nan
        return avg_episode_reward

    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using collected rollout data.
        
        Returns:
            Dictionary containing average training loss metrics.
        """
        data = self.buffer.get(self.advantages, self.returns)

        metrics_agg = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'prediction_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0
        }
        num_updates = 0
        rollout_size = len(data['rewards'])

        for epoch in range(PPO_NUM_EPOCHS):
            indices = np.random.permutation(rollout_size)

            for start in range(0, rollout_size, PPO_BATCH_SIZE):
                end = min(start + PPO_BATCH_SIZE, rollout_size)
                batch_idx = indices[start:end]

                batch_delayed_seq = data['delayed_sequences'][batch_idx]
                batch_remote_states = data['remote_states'][batch_idx]
                batch_actions = data['actions'][batch_idx]
                batch_old_log_probs = data['old_log_probs'][batch_idx]
                batch_advantages = data['advantages'][batch_idx]
                batch_returns = data['returns'][batch_idx]
                batch_true_targets = data['true_targets'][batch_idx]

                log_probs, entropy, values, predicted_targets, _ = self.policy.evaluate_actions(
                    batch_delayed_seq,
                    batch_remote_states,
                    batch_actions,
                    hidden_state=None
                )

                # Calculate Losses
                prediction_loss = F.mse_loss(predicted_targets, batch_true_targets)

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                entropy_loss = -entropy.mean()

                total_loss = (
                    PPO_LOSS_WEIGHT * (
                        actor_loss +
                        PPO_VALUE_COEF * critic_loss +
                        PPO_ENTROPY_COEF * entropy_loss
                    ) +
                    PREDICTION_LOSS_WEIGHT * prediction_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    PPO_MAX_GRAD_NORM
                )
                self.optimizer.step()

                metrics_agg['actor_loss'] += actor_loss.item()
                metrics_agg['critic_loss'] += critic_loss.item()
                metrics_agg['prediction_loss'] += prediction_loss.item()
                metrics_agg['entropy'] += entropy.mean().item()
                metrics_agg['total_loss'] += total_loss.item()
                
                with torch.no_grad():
                    log_ratio = log_probs - batch_old_log_probs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    metrics_agg['approx_kl'] += approx_kl
                    
                num_updates += 1

        return {key: val / num_updates for key, val in metrics_agg.items()}

    def train(self, total_timesteps: int):
        """Main training loop with visualization and early stopping."""
        num_updates_total = total_timesteps // PPO_ROLLOUT_STEPS

        # Initialize visualization
        self._init_tensorboard(self.checkpoint_dir)
        self.training_start_time = datetime.now()
        start_time = self.training_start_time

        for update in range(num_updates_total):
            update_start_time = time.time()

            # Collect Rollout
            avg_episode_reward = self.collect_rollout()
            collect_time = time.time() - update_start_time

            # Update Policy
            update_start_time_ppo = time.time()
            metrics = self.update_policy()
            update_time = time.time() - update_start_time_ppo

            self.num_updates += 1

            # Log Metrics (TensorBoard + JSON)
            self._log_metrics(metrics, avg_episode_reward)

            # Console Logging
            if self.num_updates % LOG_FREQ == 0:
                elapsed_time = datetime.now() - start_time
                fps = int(PPO_ROLLOUT_STEPS / (collect_time + update_time)) if (collect_time + update_time) > 0 else 0
                
                logger.info(f"\n{'─'*70}")
                logger.info(f"Update: {self.num_updates}/{num_updates_total} | "
                           f"Timesteps: {self.total_timesteps:,} | "
                           f"Elapsed: {str(elapsed_time).split('.')[0]}")
                logger.info(f"{'─'*70}")
                logger.info(f"  Avg Rollout Reward: {avg_episode_reward:.3f}")
                logger.info(f"  Actor Loss: {metrics.get('actor_loss', np.nan):.4f} | "
                           f"Critic Loss: {metrics.get('critic_loss', np.nan):.4f}")
                logger.info(f"  Prediction Loss: {metrics.get('prediction_loss', np.nan):.6f}")
                logger.info(f"  Entropy: {metrics.get('entropy', np.nan):.4f} | "
                           f"Approx KL: {metrics.get('approx_kl', np.nan):.5f}")
                logger.info(f"  Collect Time: {collect_time:.2f}s | "
                           f"Update Time: {update_time:.2f}s | FPS: {fps}")

            # Early Stopping Check
            if ENABLE_EARLY_STOPPING and self.num_updates % EARLY_STOPPING_CHECK_FREQ == 0:
                current_check_reward = avg_episode_reward
                improvement = current_check_reward - self.best_mean_reward

                if improvement >= EARLY_STOPPING_MIN_DELTA:
                    logger.info(f"\n  ✓ Early Stopping: New best reward {current_check_reward:.3f} "
                               f"(+{improvement:.3f}). Saving model.")
                    self.best_mean_reward = current_check_reward
                    self.no_improvement_count = 0
                    
                    self.best_model_path = os.path.join(self.checkpoint_dir, "best_policy_earlystop.pth")
                    self.policy.save(self.best_model_path)
                    
                    # Log to TensorBoard
                    if self.tb_writer:
                        self.tb_writer.add_scalar("train/best_reward", self.best_mean_reward, self.num_updates)
                else:
                    self.no_improvement_count += 1
                    logger.info(f"  ○ Early Stopping: No significant improvement "
                               f"({improvement:.3f} < {EARLY_STOPPING_MIN_DELTA}). "
                               f"Patience: {self.no_improvement_count}/{EARLY_STOPPING_PATIENCE}")

                if self.no_improvement_count >= EARLY_STOPPING_PATIENCE:
                    logger.warning(f"\n{'!'*70}")
                    logger.warning(f"EARLY STOPPING triggered at update {self.num_updates}")
                    logger.warning(f"No improvement for {EARLY_STOPPING_PATIENCE} checks.")
                    logger.warning(f"{'!'*70}\n")
                    break

            # Periodic Checkpoint Saving
            if self.num_updates % SAVE_FREQ == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"policy_update_{self.num_updates}.pth"
                )
                self.policy.save(checkpoint_path)
                logger.info(f"  💾 Checkpoint saved: {checkpoint_path}")

        # Final Model Saving
        logger.info(f"\n{'='*70}")
        if not ENABLE_EARLY_STOPPING or self.no_improvement_count < EARLY_STOPPING_PATIENCE:
            final_path = os.path.join(self.checkpoint_dir, "final_policy.pth")
            self.policy.save(final_path)
            logger.info(f"Training Finished")
            logger.info(f"Final model saved: {final_path}")
        else:
            logger.info(f"Training Stopped Early")
            logger.info(f"Best model saved: {self.best_model_path}")

        total_training_time = datetime.now() - start_time
        logger.info(f"Total Training Time: {str(total_training_time).split('.')[0]}")
        logger.info(f"{'='*70}\n")
        
        # Save metrics and close loggers
        self._save_metrics_json()
        self._save_training_summary()
        
        if self.tb_writer:
            self.tb_writer.close()
            logger.info("TensorBoard writer closed")
        
        logger.info(f"All training data saved to: {self.checkpoint_dir}")