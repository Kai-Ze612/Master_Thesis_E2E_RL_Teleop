"""
Custom recurrent PPO policy for end to end training.

This script defines the backward training loop and optimization process for the recurrent PPO agent.

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
from typing import Dict, List, Optional, Tuple
import time
import json

# stable baselines3 imports
from stable_baselines3.common.vec_env import VecEnv  # make vector environment
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
    EARLY_STOPPING_CHECK_FREQ,
    NUM_ENVIRONMENTS
)

logger = logging.getLogger(__name__)


class RecurrentPPOTrainer:
    
    def __init__(self, env: VecEnv):
        """Initialize the Recurrent PPO Trainer."""
        
        # Initialize environment
        self.env = env
        self.num_envs = env.num_envs
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize policy
        self.policy = RecurrentPPOPolicy().to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=PPO_LEARNING_RATE, eps=1e-5)

        # Rollout buffer - we will divide PPO_ROLLOUT_STEPS by num_envs
        # Which means we create separate buffers for each env (parallel env for data collection)
        if PPO_ROLLOUT_STEPS % self.num_envs != 0:
            logger.warning(f"PPO_ROLLOUT_STEPS ({PPO_ROLLOUT_STEPS}) not divisible by NUM_ENVIRONMENTS ({self.num_envs}).")
            buffer_size = PPO_ROLLOUT_STEPS # adjust the buffer size accordingly (not ideal)
        else:
             buffer_size = PPO_ROLLOUT_STEPS

        # Initialize rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size
        )

        self.total_timesteps = 0
        self.num_updates = 0
        self.checkpoint_dir = CHECKPOINT_DIR
        
        # Initialize tensorboard writer
        self.tb_writer: Optional[SummaryWriter] = None
        self.metrics_history = { 'steps': [], 'timestamps': [], 'rewards': [], 'actor_losses': [], 'critic_losses': [],
                               'prediction_losses': [], 'entropies': [], 'approx_kls': [], 'total_losses': [] }
        self.training_start_time = None

        # for early stopping
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0

    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        tb_dir = os.path.join(self.checkpoint_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=120)
        logger.info(f"  Tensorboard login: http://localhost:6006/")

    def _log_metrics(self, metrics: Dict[str, float], avg_reward: float):
        """Log metrics to both TensorBoard and JSON storage."""
        
        step = self.num_updates
        if np.isnan(avg_reward): avg_reward = 0.0 # Handle potential NaN early on

        elapsed = (datetime.now() - self.training_start_time).total_seconds()
        self.metrics_history['steps'].append(step)
        self.metrics_history['timestamps'].append(elapsed)
        self.metrics_history['rewards'].append(float(avg_reward)) # Ensure float
        self.metrics_history['actor_losses'].append(float(metrics.get('actor_loss', 0.0)))
        self.metrics_history['critic_losses'].append(float(metrics.get('critic_loss', 0.0)))
        self.metrics_history['prediction_losses'].append(float(metrics.get('prediction_loss', 0.0)))
        self.metrics_history['entropies'].append(float(metrics.get('entropy', 0.0)))
        self.metrics_history['approx_kls'].append(float(metrics.get('approx_kl', 0.0)))
        self.metrics_history['total_losses'].append(float(metrics.get('total_loss', 0.0)))

        if self.tb_writer:
            self.tb_writer.add_scalar('train/avg_reward', avg_reward, step)
            self.tb_writer.add_scalars('losses', {
                'actor': metrics.get('actor_loss', 0.0),
                'critic': metrics.get('critic_loss', 0.0),
                'prediction': metrics.get('prediction_loss', 0.0),
                'total': metrics.get('total_loss', 0.0)
            }, step)
            self.tb_writer.add_scalar('train/entropy', metrics.get('entropy', 0.0), step)
            self.tb_writer.add_scalar('train/approx_kl', metrics.get('approx_kl', 0.0), step)
            if step % (LOG_FREQ * 2) == 0: self.tb_writer.flush()

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
            'metrics': self.metrics_history }
        try:
             with open(metrics_file, 'w') as f: json.dump(output, f, indent=2)
             logger.info(f"Training metrics saved to: {metrics_file}")
        except Exception as e:
             logger.error(f"Failed to save metrics JSON: {e}")

    def _save_training_summary(self):
        """Save human-readable training summary."""
        summary_file = os.path.join(self.checkpoint_dir, "training_summary.txt")
        try:
            with open(summary_file, 'w') as f:
                f.write("="*70 + "\nTraining Summary\n" + "="*70 + "\n\n")
                if self.training_start_time:
                    duration = (datetime.now() - self.training_start_time).total_seconds()
                    f.write(f"Start Time: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Duration: {duration / 3600:.2f} hours ({duration / 60:.1f} minutes)\n\n")
                f.write(f"Total Updates: {len(self.metrics_history['steps'])}\n")
                f.write(f"Total Timesteps: {self.total_timesteps:,}\n\n")
                if self.metrics_history['rewards']:
                    rewards = [r for r in self.metrics_history['rewards'] if not np.isnan(r)] # Filter NaNs
                    if rewards:
                        f.write("Reward Statistics:\n")
                        f.write(f"  Initial: {rewards[0]:.3f}\n")
                        f.write(f"  Final: {rewards[-1]:.3f}\n")
                        f.write(f"  Best: {max(rewards):.3f}\n")
                        f.write(f"  Mean: {sum(rewards)/len(rewards):.3f}\n")
                        f.write(f"  Std: {np.std(rewards):.3f}\n\n")
                if self.metrics_history['prediction_losses']:
                    pred_losses = [l for l in self.metrics_history['prediction_losses'] if not np.isnan(l)]
                    if pred_losses:
                        f.write("Prediction Loss (LSTM):\n")
                        f.write(f"  Initial: {pred_losses[0]:.6f}\n")
                        f.write(f"  Final: {pred_losses[-1]:.6f}\n")
                        f.write(f"  Best: {min(pred_losses):.6f}\n\n")
                if self.best_mean_reward > -np.inf:
                    f.write(f"Best Reward Achieved (Early Stopping): {self.best_mean_reward:.3f}\n")
                f.write("\n" + "="*70 + "\n")
            logger.info(f"Training summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")

    def collect_rollout(self) -> float:
        """Collect rollout data from multiple environments in parallel."""

        # Reset buffer
        self.buffer.reset()

        # Reset VecEnv and get initial observations
        # It aims to batch the initial observations from all parallel environments
        try:
            current_obs_batch = self.env.reset()
            if not isinstance(current_obs_batch, np.ndarray):
                 current_obs_batch = np.array(current_obs_batch)
            if current_obs_batch.shape[0] != self.num_envs:
                 raise ValueError(f"Reset obs batch size {current_obs_batch.shape[0]} != num_envs {self.num_envs}")
        except Exception as e:
            logger.error(f"Error during VecEnv reset: {e}", exc_info=True)
            return -np.inf # Signal error

        # Initialize hidden states (one for each env)
        hidden_state: HiddenStateType = self.policy.init_hidden_state(batch_size=self.num_envs, device=self.device)
        episode_rewards_list: List[float] = [] # Store completed episode rewards
        current_episode_rewards = np.zeros(self.num_envs, dtype=np.float32)

        # Determine number of steps PER ENV
        # Assuming PPO_ROLLOUT_STEPS is TOTAL steps across all envs
        num_steps_per_env = PPO_ROLLOUT_STEPS // self.num_envs
        if PPO_ROLLOUT_STEPS % self.num_envs != 0:
            logger.warning(f"Rollout steps {PPO_ROLLOUT_STEPS} not divisible by num_envs {self.num_envs}. Using {num_steps_per_env} steps per env.")

        # Rollout Loop
        for step in range(num_steps_per_env): # Loop for steps per env
            # Get data from VecEnv using env_method
            try:
                # These return LISTS (one item per env)
                delayed_buffers_list = self.env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
                remote_states_list = self.env.env_method("get_remote_state")
                true_targets_list = self.env.env_method("get_true_current_target")

                # Stack the lists into batches (num_envs, ...)
                delayed_sequences_batch = np.array([buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2) for buf in delayed_buffers_list])
                remote_states_batch = np.array(remote_states_list)
                true_targets_batch = np.array(true_targets_list)

            except Exception as e:
                 logger.error(f"Error getting batched env data at step {step}: {e}", exc_info=True)
                 return -np.inf # Signal error

            # Convert batches to tensors
            delayed_sequences_t = torch.FloatTensor(delayed_sequences_batch).to(self.device)
            remote_states_t = torch.FloatTensor(remote_states_batch).to(self.device)

            # Get actions (batch) from policy
            with torch.no_grad():
                # Policy now operates on batch dimension (num_envs)
                action_t, log_prob_t, value_t, predicted_target_t, new_hidden_state = self.policy.get_action(
                    delayed_sequences_t,
                    remote_states_t,
                    hidden_state, # Pass the batched hidden state
                    deterministic=False
                )

            # Convert tensors to numpy batches
            actions_np = action_t.cpu().numpy()
            predicted_targets_np = predicted_target_t.cpu().numpy()
            log_probs_np = log_prob_t.cpu().numpy() if log_prob_t is not None else np.zeros(self.num_envs)
            values_np = value_t.cpu().numpy().flatten()

            # Set predictions for EACH env INDIVIDUALLY
            try:
                for i in range(self.num_envs):
                    # Call env_method for each environment 'i', passing only its prediction
                    self.env.env_method(
                        "set_predicted_target",
                        predicted_targets_np[i], # Pass the specific prediction for this env
                        indices=[i]              # Target only environment 'i'
                    )
            except Exception as e:
                logger.error(f"Error setting predicted targets in VecEnv: {e}", exc_info=True)
                return -np.inf # Signal error

            # Step VecEnv (takes batch of actions, returns batches)
            try:
                next_obs_batch, rewards_batch, dones_batch, infos_batch = self.env.step(actions_np)
            except Exception as e:
                logger.error(f"Error during VecEnv step {step}: {e}", exc_info=True)
                return -np.inf # Signal error

            # Store transitions in buffer (iterate through the batch)
            for i in range(self.num_envs):
                self.buffer.add(
                    delayed_sequence=delayed_sequences_batch[i],
                    remote_state=remote_states_batch[i],
                    action=actions_np[i],
                    log_prob=log_probs_np[i],
                    value=values_np[i],
                    reward=float(rewards_batch[i]),
                    done=bool(dones_batch[i]), # Use done from VecEnv step
                    predicted_target=predicted_targets_np[i],
                    true_target=true_targets_batch[i]
                )

            current_episode_rewards += rewards_batch
            self.total_timesteps += self.num_envs

            # Update hidden state (already batched)
            hidden_state = new_hidden_state

            # Handle episode ends (per environment)
            for i in range(self.num_envs):
                if dones_batch[i]:
                    # Log and reset reward counter
                    episode_rewards_list.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0.0

                    # Reset LSTM state for finished environments
                    # Ensure hidden_state is mutable if it's a tuple
                    if isinstance(hidden_state, tuple):
                        h, c = hidden_state
                        # Make copies to avoid modifying tensor in-place issues if needed by graph
                        h = h.clone()
                        c = c.clone()
                        h[:, i, :] = 0.0 # Reset hidden state for env i
                        c[:, i, :] = 0.0 # Reset cell state for env i
                        hidden_state = (h, c)
                    elif isinstance(hidden_state, torch.Tensor): # Assuming tensor for GRU
                        hidden_state = hidden_state.clone()
                        hidden_state[:, i, :] = 0.0

                    # Note: VecEnv handles resetting the actual environment state internally
                    # `next_obs_batch` already contains the reset observation for env i
            # ----------------------------------------------

            # Update current observation batch for the next loop iteration (redundant if obs aren't used next loop)
            # current_obs_batch = next_obs_batch

        # --- Verify buffer has data ---
        # Use the len() dunder method of the buffer
        if len(self.buffer) == 0:
             logger.error("Buffer is empty after collection! This should not happen.")
             return -np.inf # Signal error

        # --- Calculate last value for GAE (needs batching) ---
        try:
            # Need the observations corresponding to the *next* state after the loop finished
            # We already have next_obs_batch from the last step of the loop
            # We need the corresponding delayed sequences and remote states for that next step
            # This requires calling env_method again AFTER the loop
            last_delayed_buffers = self.env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
            last_remote_states = self.env.env_method("get_remote_state")
            last_delayed_sequences = np.array([buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2) for buf in last_delayed_buffers])
            last_remote_states_np = np.array(last_remote_states)

            last_delayed_sequences_t = torch.FloatTensor(last_delayed_sequences).to(self.device)
            last_remote_states_t = torch.FloatTensor(last_remote_states_np).to(self.device)

            with torch.no_grad():
                # Use the final hidden state from the loop
                _, _, last_values_t, _, _ = self.policy.forward(
                    last_delayed_sequences_t,
                    last_remote_states_t,
                    hidden_state
                )

            # --- More robust GAE: Use last_values only for non-terminated envs ---
            # We need the 'dones' from the last step of the loop (`dones_batch`)
            last_values_np = last_values_t.cpu().numpy().flatten()
            # If an env was 'done' at the last step, its next value is 0 for GAE calculation
            # We pass the calculated values, GAE handles the 'done' flag internally
            # For the buffer's compute_returns_and_advantages, it needs a *single* value if the
            # whole rollout didn't end. This is tricky with VecEnv.
            # Option 1: Pass the mean value (simple approximation)
            # last_value_for_buffer = last_values_np.mean()
            # Option 2: Pass all values and modify buffer (complex)
            # Option 3: Assume the *buffer* handles dones correctly and pass the value
            #           of the *next state* after the last buffer entry. If the LAST step
            #           in the buffer was done, the corresponding last_value doesn't matter.
            #           Let's assume the buffer handles dones in GAE and pass the mean.
            last_value_for_buffer = last_values_np.mean()


        except Exception as e:
            logger.error(f"Error getting last value for GAE: {e}", exc_info=True)
            return -np.inf # Signal error

        # --- Compute Advantages and Returns ---
        try:
            # Pass the single (approximated) last value
            self.advantages, self.returns = self.buffer.compute_returns_and_advantages(
                 last_value_for_buffer, PPO_GAMMA, PPO_GAE_LAMBDA
             )
        except Exception as e:
            logger.error(f"Error computing GAE: {e}", exc_info=True)
            return -np.inf


        avg_episode_reward = np.mean(episode_rewards_list) if episode_rewards_list else np.nan
        return avg_episode_reward

    def update_policy(self) -> Dict[str, float]:
        """Update policy using collected rollout data."""
        try:
             # Ensure the buffer check uses the dunder len method
             if len(self.buffer) == 0:
                  raise ValueError("Buffer is empty, cannot get data.")
             data = self.buffer.get_policy_data(self.advantages, self.returns)
        except ValueError as e: # Catch the "Buffer is empty" error specifically
             logger.error(f"Cannot update policy: {e}")
             # Return empty metrics or NaN to signal failure
             return { 'actor_loss': np.nan, 'critic_loss': np.nan, 'prediction_loss': np.nan,
                      'entropy': np.nan, 'total_loss': np.nan, 'approx_kl': np.nan }

        metrics_agg = { 'actor_loss': 0.0, 'critic_loss': 0.0, 'prediction_loss': 0.0,
                        'entropy': 0.0, 'total_loss': 0.0, 'approx_kl': 0.0 }
                
        num_minibatch_updates = 0 # Use a different counter name
        rollout_size = len(data['advantages']) # Correctly uses 'advantages'

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
                    hidden_state=None # Assuming stateless evaluation during update
                )

                # Calculate Losses
                prediction_loss = F.mse_loss(predicted_targets, batch_true_targets)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # Ensure values has the correct shape for mse_loss (e.g., [batch_size])
                critic_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                entropy_loss = -entropy.mean()
                total_loss = ( PPO_LOSS_WEIGHT * ( actor_loss + PPO_VALUE_COEF * critic_loss + PPO_ENTROPY_COEF * entropy_loss ) +
                               PREDICTION_LOSS_WEIGHT * prediction_loss )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), PPO_MAX_GRAD_NORM)
                self.optimizer.step()

                # Aggregate metrics
                metrics_agg['actor_loss'] += actor_loss.item()
                metrics_agg['critic_loss'] += critic_loss.item()
                metrics_agg['prediction_loss'] += prediction_loss.item()
                metrics_agg['entropy'] += entropy.mean().item()
                metrics_agg['total_loss'] += total_loss.item()
                with torch.no_grad():
                     log_ratio = log_probs - batch_old_log_probs
                     approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                     metrics_agg['approx_kl'] += approx_kl.item() # Use .item()
                num_minibatch_updates += 1

        # Average metrics over all minibatch updates
        return {key: val / num_minibatch_updates for key, val in metrics_agg.items()}

    def train(self, total_timesteps: int):
        """Main training loop."""
        
        # Total number of updates
        num_updates_total = total_timesteps // PPO_ROLLOUT_STEPS

        # Initialize TensorBoard
        self._init_tensorboard()
        
        # Initialize training time
        self.training_start_time = datetime.now()
        start_time = self.training_start_time
        
        # Set best model path relative to the specific run's checkpoint dir
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_policy_earlystop.pth")

        for update in range(num_updates_total):
            update_start_time = time.time()

            # Collect Rollout
            avg_episode_reward = self.collect_rollout()
            if np.isinf(avg_episode_reward):
                 logger.error("Rollout collection failed, stopping training.")
                 break
            collect_time = time.time() - update_start_time

            # Update Policy
            update_start_time_ppo = time.time()
            metrics = self.update_policy()
            if any(np.isnan(v) for v in metrics.values()): # Check for error signal
                 logger.error("Policy update failed (NaN metrics), stopping training.")
                 break
            update_time = time.time() - update_start_time_ppo

            self.num_updates += 1

            # Log Metrics
            self._log_metrics(metrics, avg_episode_reward)

            # Console Logging
            if self.num_updates % LOG_FREQ == 0:
                elapsed_time = datetime.now() - start_time
                total_cycle_time = collect_time + update_time
                fps = int(PPO_ROLLOUT_STEPS / total_cycle_time) if total_cycle_time > 0 else 0

                logger.info(f"\n{'─'*70}")
                # Display update number relative to total calculated updates
                logger.info(f"Update: {self.num_updates}/{num_updates_total} | Timesteps: {self.total_timesteps:,} | Elapsed: {str(elapsed_time).split('.')[0]}")
                logger.info(f"{'─'*70}")
                # Handle potential NaN reward early in training
                reward_str = f"{avg_episode_reward:.3f}" if not np.isnan(avg_episode_reward) else "NaN"
                logger.info(f"  Avg Rollout Reward: {reward_str}")
                logger.info(f"  Actor Loss: {metrics.get('actor_loss', np.nan):.4f} | Critic Loss: {metrics.get('critic_loss', np.nan):.4f}")
                logger.info(f"  Prediction Loss: {metrics.get('prediction_loss', np.nan):.6f}")
                logger.info(f"  Entropy: {metrics.get('entropy', np.nan):.4f} | Approx KL: {metrics.get('approx_kl', np.nan):.5f}")
                logger.info(f"  Collect Time: {collect_time:.2f}s | Update Time: {update_time:.2f}s | FPS: {fps}")

            # Early Stopping Check
            if ENABLE_EARLY_STOPPING and self.num_updates % EARLY_STOPPING_CHECK_FREQ == 0:
                 current_check_reward = avg_episode_reward
                 # Ensure reward is valid before checking
                 if not np.isnan(current_check_reward):
                     improvement = current_check_reward - self.best_mean_reward
                     if improvement >= EARLY_STOPPING_MIN_DELTA:
                         logger.info(f"\n Early Stopping: New best reward {current_check_reward:.3f} (+{improvement:.3f}). Saving model.")
                         self.best_mean_reward = current_check_reward
                         self.no_improvement_count = 0
                         # Save to the dynamically set best_model_path
                         self.policy.save(self.best_model_path)
                         if self.tb_writer: self.tb_writer.add_scalar("train/best_reward", self.best_mean_reward, self.num_updates)
                     else:
                         self.no_improvement_count += 1
                         logger.info(f" Early Stopping: No improvement ({improvement:.3f} < {EARLY_STOPPING_MIN_DELTA}). Patience: {self.no_improvement_count}/{EARLY_STOPPING_PATIENCE}")
                     if self.no_improvement_count >= EARLY_STOPPING_PATIENCE:
                         logger.warning(f"\n{'!'*70}\nEARLY STOPPING triggered at update {self.num_updates}\n{'!'*70}\n")
                         break # Exit the training loop
                 else:
                     logger.warning("Skipping early stopping check due to NaN reward in this update cycle.")


            # Periodic Checkpoint Saving
            if self.num_updates % SAVE_FREQ == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"policy_update_{self.num_updates}.pth")
                try:
                    self.policy.save(checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")

        # Final actions after loop finishes or breaks
        logger.info(f"\n{'='*70}")
        final_path = os.path.join(self.checkpoint_dir, "final_policy.pth")

        # Check if loop finished normally or stopped early
        training_complete = (self.num_updates >= num_updates_total) and (not ENABLE_EARLY_STOPPING or self.no_improvement_count < EARLY_STOPPING_PATIENCE)

        if training_complete:
            logger.info(f"Training Finished ({self.num_updates}/{num_updates_total} updates).")
            try:
                self.policy.save(final_path)
                logger.info(f"Final model saved: {final_path}")
            except Exception as e:
                 logger.error(f"Failed to save final model: {e}")
        elif ENABLE_EARLY_STOPPING and self.no_improvement_count >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Training Stopped Early due to early stopping criteria.")
            logger.info(f"Best model already saved: {self.best_model_path}")
            # Optionally save the final state anyway
            # self.policy.save(final_path); logger.info(f"Final (non-best) model saved: {final_path}")
        else: # Likely stopped by error or KeyboardInterrupt handled in train_agent
             logger.info(f"Training loop exited before completion ({self.num_updates}/{num_updates_total} updates).")
             # Final save might have happened in exception handler in train_agent

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