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
from typing import Dict, List
import time

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
 
    def __init__(self, env):
        
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize policy
        self.policy = RecurrentPPOPolicy().to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=PPO_LEARNING_RATE,
            eps=1e-5  # Adam epsilon for numerical stability
        )

        # Rollout buffer
        # RolloutBuffer is for storing training data. It stores all experiences the agent collected during environment interaction.
        self.buffer = RolloutBuffer(
            buffer_size=PPO_ROLLOUT_STEPS,
            device=self.device
        )

        # Training state
        self.total_timesteps = 0
        self.num_updates = 0
        self.checkpoint_dir = CHECKPOINT_DIR

        # Early Stopping Attributes
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_policy_earlystop.pth")

    def collect_rollout(self) -> float:
        """
        Collect one rollout (PPO_ROLLOUT_STEPS) of experience.
        
        Key Flow:
            1. Get delayed observations from environment
            2. Policy predicts current state using LSTM
            3. Policy outputs torque compensation action
            4. Pass predicted state to environment via set_predicted_target()
            5. Environment calculates dense reward (prediction + tracking)
            6. Store experience in buffer
            7. Manage LSTM hidden state (carry forward, reset on episode end)
        
        Returns:
            Average episode reward encountered during the rollout.
        """
        # Reset buffer at start of rollout
        self.buffer.reset()
        
        # Reset environment
        try:
            obs_dict = self.env.reset()
            # Handle tuple vs single array return
            if isinstance(obs_dict, tuple):
                initial_obs = obs_dict[0]  # (obs, info)
            else:
                initial_obs = obs_dict
                
            # Verify observation type
            if not isinstance(initial_obs, np.ndarray):
                logger.warning(f"env.reset() did not return NumPy array. Type: {type(initial_obs)}")
                
        except Exception as e:
            logger.error(f"Error during environment reset: {e}", exc_info=True)
            return -np.inf

        # Initialize LSTM hidden state for the beginning of the rollout
        # Batch size is 1 since we collect step-by-step
        hidden_state: HiddenStateType = self.policy.init_hidden_state(batch_size=1, device=self.device)

        episode_rewards: List[float] = []
        current_episode_reward: float = 0.0
        steps_collected: int = 0

        while steps_collected < PPO_ROLLOUT_STEPS:
            # Get observation from environment
            # Delayed buffer: shape (seq_len * features,) needs reshaping for LSTM
            delayed_buffer_flat = self.env.get_delayed_target_buffer(RNN_SEQUENCE_LENGTH)
            # Reshape: (RNN_SEQUENCE_LENGTH * N_JOINTS * 2,) → (RNN_SEQUENCE_LENGTH, N_JOINTS * 2)
            delayed_sequence = delayed_buffer_flat.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2)

            remote_state = self.env.get_remote_state()  # Shape (14,) = [q, qd]
            true_target = self.env.get_true_current_target()  # Shape (14,) = [q_true, qd_true]

            # Convert to tensors (add batch dimension)
            delayed_sequence_t = torch.FloatTensor(delayed_sequence).unsqueeze(0).to(self.device)
            remote_state_t = torch.FloatTensor(remote_state).unsqueeze(0).to(self.device)

            # Get action from policy
            with torch.no_grad():
                # LSTM predicts state → Actor outputs action
                action_t, log_prob_t, value_t, predicted_target_t, new_hidden_state = self.policy.get_action(
                    delayed_sequence_t,
                    remote_state_t,
                    hidden_state,
                    deterministic=False  # Sample during training
                )

            # Convert tensors to numpy for storage and env interaction
            action_np = action_t.cpu().numpy().squeeze(axis=0)  # Remove batch dim → (7,)
            predicted_target_np = predicted_target_t.cpu().numpy().squeeze(axis=0)  # → (14,)
            log_prob_val = log_prob_t.item() if log_prob_t is not None else 0.0
            value_val = value_t.item()

            # Pass prediction to environment BEFORE stepping
            # Environment uses this to calculate prediction reward component
            self.env.set_predicted_target(predicted_target_np)

            # step environment
            try:
                next_obs_tuple = self.env.step(action_np)
                
                # Unpack based on gym version
                if len(next_obs_tuple) == 5:
                    next_obs, reward, terminated, truncated, info = next_obs_tuple
                    done = terminated or truncated
                elif len(next_obs_tuple) == 4:  # Older gym interface
                    next_obs, reward, done, info = next_obs_tuple
                else:
                    raise ValueError(f"Unexpected env.step() return: {len(next_obs_tuple)} values")
                    
            except Exception as e:
                logger.error(f"Error during environment step {steps_collected}: {e}", exc_info=True)
                break  # Stop rollout on error

            # Store transition in buffer
            self.buffer.add(
                delayed_sequence=delayed_sequence,  # Store unbatched sequence
                remote_state=remote_state,
                action=action_np,
                log_prob=log_prob_val,
                value=value_val,
                reward=float(reward),  # Use reward from environment
                done=bool(done),
                predicted_target=predicted_target_np,
                true_target=true_target
            )

            current_episode_reward += float(reward)
            self.total_timesteps += 1
            steps_collected += 1

            # Update LSTM hidden state (carry forward across time)
            hidden_state = new_hidden_state

            # Handle episode end
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                
                # Reset environment and LSTM hidden state
                reset_output = self.env.reset()
                if isinstance(reset_output, tuple):
                    obs = reset_output[0]
                else:
                    obs = reset_output
                    
                # Reset hidden state on episode boundary
                hidden_state = self.policy.init_hidden_state(batch_size=1, device=self.device)

        # Calculate last value for GAE bootstrap
        # Get the state corresponding to the very last step
        last_delayed_buffer = self.env.get_delayed_target_buffer(RNN_SEQUENCE_LENGTH)
        last_remote_state = self.env.get_remote_state()
        last_delayed_sequence = last_delayed_buffer.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2)
        last_delayed_sequence_t = torch.FloatTensor(last_delayed_sequence).unsqueeze(0).to(self.device)
        last_remote_state_t = torch.FloatTensor(last_remote_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Run forward pass to get value of the state *after* the last step
            _, _, _, last_value_t, _ = self.policy.forward(
                last_delayed_sequence_t,
                last_remote_state_t,
                hidden_state  # Use the hidden state *after* the last step
            )
        last_value = last_value_t.item()

        # --- Compute Advantages and Returns using GAE ---
        self.advantages, self.returns = self.buffer.compute_returns_and_advantages(
            last_value, PPO_GAMMA, PPO_GAE_LAMBDA
        )

        avg_episode_reward = np.mean(episode_rewards) if episode_rewards else np.nan
        return avg_episode_reward

    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using collected rollout data stored in the buffer.
        Performs multiple PPO epochs over mini-batches.
        
        Training Objective:
            Total Loss = PPO_LOSS_WEIGHT * (Actor + Value + Entropy) 
                       + PREDICTION_LOSS_WEIGHT * State_Prediction
        """
        
        # Retrieve data from buffer (includes advantages and returns)
        data = self.buffer.get(self.advantages, self.returns)

        # Training metrics accumulators
        metrics_agg = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'prediction_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0
        }
        num_updates = 0
        rollout_size = len(data['rewards'])  # Actual rollout size

        # PPO update epochs
        for epoch in range(PPO_NUM_EPOCHS):
            # Generate random indices for mini-batches
            indices = np.random.permutation(rollout_size)

            for start in range(0, rollout_size, PPO_BATCH_SIZE):
                end = min(start + PPO_BATCH_SIZE, rollout_size)
                batch_idx = indices[start:end]

                # Prepare mini-batch data
                batch_delayed_seq = data['delayed_sequences'][batch_idx]
                batch_remote_states = data['remote_states'][batch_idx]
                batch_actions = data['actions'][batch_idx]
                batch_old_log_probs = data['old_log_probs'][batch_idx]
                batch_advantages = data['advantages'][batch_idx]
                batch_returns = data['returns'][batch_idx]
                batch_true_targets = data['true_targets'][batch_idx]

                # Evaluate actions with current policy
                # Note: We reset hidden state to None for each mini-batch
                # This is an approximation - a more rigorous approach would store hidden states
                log_probs, entropy, values, predicted_targets, _ = self.policy.evaluate_actions(
                    batch_delayed_seq,
                    batch_remote_states,
                    batch_actions,
                    hidden_state=None  # Reset for each mini-batch
                )

                # Calculate individual loss components
                # Prediction Loss : MSE between predicted and true current local robot state
                prediction_loss = F.mse_loss(predicted_targets, batch_true_targets)

                # PPO Actor Loss : Clipped Surrogate Objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # PPO Critic Loss : MSE between predicted value and computed return
                critic_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Entropy Bonus (For Exploration)
                entropy_loss = -entropy.mean()

                # Total Loss (End-to-End Weighted Sum)
                total_loss = (
                    PPO_LOSS_WEIGHT * (
                        actor_loss +
                        PPO_VALUE_COEF * critic_loss +
                        PPO_ENTROPY_COEF * entropy_loss
                    ) +
                    PREDICTION_LOSS_WEIGHT * prediction_loss
                )

                # Optimization Step
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients for training stability
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    PPO_MAX_GRAD_NORM
                )
                self.optimizer.step()

                # Aggregate Metrics
                metrics_agg['actor_loss'] += actor_loss.item()
                metrics_agg['critic_loss'] += critic_loss.item()
                metrics_agg['prediction_loss'] += prediction_loss.item()
                metrics_agg['entropy'] += entropy.mean().item()  # Positive entropy value
                metrics_agg['total_loss'] += total_loss.item()
                
                # Calculate approximate KL divergence (useful for debugging PPO)
                with torch.no_grad():
                    log_ratio = log_probs - batch_old_log_probs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    metrics_agg['approx_kl'] += approx_kl
                    
                num_updates += 1

        # Return average metrics over all mini-batches and epochs
        return {key: val / num_updates for key, val in metrics_agg.items()}

    def train(self, total_timesteps: int):
        """Main training loop with early stopping."""
        
        num_updates_total = total_timesteps // PPO_ROLLOUT_STEPS

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Recurrent-PPO Training")
        logger.info(f"{'='*60}")
        logger.info(f"  Total timesteps: {total_timesteps:,}")
        logger.info(f"  Updates planned: {num_updates_total:,}")
        logger.info(f"  Steps per update: {PPO_ROLLOUT_STEPS}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"{'='*60}\n")

        start_time = datetime.now()

        for update in range(num_updates_total):
            update_start_time = time.time()

            # --- Collect Rollout ---
            avg_episode_reward = self.collect_rollout()
            collect_time = time.time() - update_start_time

            # --- Update Policy ---
            update_start_time_ppo = time.time()
            metrics = self.update_policy()
            update_time = time.time() - update_start_time_ppo

            self.num_updates += 1  # Increment update counter

            # --- Logging ---
            if self.num_updates % LOG_FREQ == 0:
                elapsed_time = datetime.now() - start_time
                fps = int(PPO_ROLLOUT_STEPS / (collect_time + update_time)) if (collect_time + update_time) > 0 else 0
                
                logger.info(f"\n{'─'*60}")
                logger.info(f"Update: {self.num_updates}/{num_updates_total} | "
                           f"Timesteps: {self.total_timesteps:,} | "
                           f"Elapsed: {str(elapsed_time).split('.')[0]}")
                logger.info(f"{'─'*60}")
                logger.info(f"  Avg Rollout Reward: {avg_episode_reward:.3f}")
                logger.info(f"  Actor Loss: {metrics.get('actor_loss', np.nan):.4f} | "
                           f"Critic Loss: {metrics.get('critic_loss', np.nan):.4f}")
                logger.info(f"  Prediction Loss: {metrics.get('prediction_loss', np.nan):.6f}")
                logger.info(f"  Entropy: {metrics.get('entropy', np.nan):.4f} | "
                           f"Approx KL: {metrics.get('approx_kl', np.nan):.5f}")
                logger.info(f"  Collect Time: {collect_time:.2f}s | "
                           f"Update Time: {update_time:.2f}s | FPS: {fps}")

            # --- Early Stopping Check ---
            if ENABLE_EARLY_STOPPING and self.num_updates % EARLY_STOPPING_CHECK_FREQ == 0:
                current_check_reward = avg_episode_reward
                improvement = current_check_reward - self.best_mean_reward

                if improvement >= EARLY_STOPPING_MIN_DELTA:
                    logger.info(f"\n  ✓ Early Stopping: New best reward {current_check_reward:.3f} "
                               f"(+{improvement:.3f}). Saving model.")
                    self.best_mean_reward = current_check_reward
                    self.no_improvement_count = 0
                    
                    # Update best model path relative to current checkpoint_dir
                    self.best_model_path = os.path.join(self.checkpoint_dir, "best_policy_earlystop.pth")
                    self.policy.save(self.best_model_path)
                else:
                    self.no_improvement_count += 1
                    logger.info(f"  ○ Early Stopping: No significant improvement "
                               f"({improvement:.3f} < {EARLY_STOPPING_MIN_DELTA}). "
                               f"Patience: {self.no_improvement_count}/{EARLY_STOPPING_PATIENCE}")

                if self.no_improvement_count >= EARLY_STOPPING_PATIENCE:
                    logger.warning(f"\n{'!'*60}")
                    logger.warning(f"EARLY STOPPING triggered at update {self.num_updates}")
                    logger.warning(f"No improvement for {EARLY_STOPPING_PATIENCE} checks.")
                    logger.warning(f"{'!'*60}\n")
                    break  # Exit the training loop

            # --- Periodic Checkpoint Saving ---
            if self.num_updates % SAVE_FREQ == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"policy_update_{self.num_updates}.pth"
                )
                self.policy.save(checkpoint_path)
                logger.info(f"  💾 Checkpoint saved: {checkpoint_path}")

        # --- Final Model Saving ---
        logger.info(f"\n{'='*60}")
        if not ENABLE_EARLY_STOPPING or self.no_improvement_count < EARLY_STOPPING_PATIENCE:
            # Training completed fully or stopping wasn't enabled
            final_path = os.path.join(self.checkpoint_dir, "final_policy.pth")
            self.policy.save(final_path)
            logger.info(f"Training Finished")
            logger.info(f"Final model saved: {final_path}")
        else:
            # Stopped early, best model already saved
            logger.info(f"Training Stopped Early")
            logger.info(f"Best model saved: {self.best_model_path}")

        total_training_time = datetime.now() - start_time
        logger.info(f"Total Training Time: {str(total_training_time).split('.')[0]}")
        logger.info(f"{'='*60}\n")