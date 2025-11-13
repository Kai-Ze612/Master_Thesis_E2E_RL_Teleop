"""
Custom Model-Based SAC (Soft Actor-Critic) Training Algorithm.

Pipeline:
1. Load pre-trained LSTM State Estimator.
2. Freeze the State Estimator parameters.
3. Train Actor and Critic networks using SAC.
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
from copy import deepcopy
from collections import deque

from stable_baselines3.common.vec_env import VecEnv 

from torch.utils.tensorboard import SummaryWriter

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import (
    StateEstimator, Actor, Critic
)

from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_CONTROL_FREQ,
    RNN_SEQUENCE_LENGTH,
    SAC_LEARNING_RATE,
    ALPHA_LEARNING_RATE,
    SAC_GAMMA,
    SAC_TAU,
    SAC_BATCH_SIZE,
    SAC_BUFFER_SIZE,
    SAC_TARGET_ENTROPY,
    SAC_UPDATES_PER_STEP,
    SAC_START_STEPS,
    SAC_VAL_FREQ,
    SAC_VAL_EPISODES,
    SAC_EARLY_STOPPING_PATIENCE,
    LOG_FREQ,
    SAVE_FREQ,
    CHECKPOINT_DIR_LSTM,
    CHECKPOINT_DIR_RL,
    NUM_ENVIRONMENTS
)

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """
    to store all collected experience for SAC training.
    
    pipeline:
    1. Add experience
    2. Sample mini-batches for training
    """
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.seq_len = RNN_SEQUENCE_LENGTH
        self.state_dim = N_JOINTS * 2
        self.action_dim = N_JOINTS
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.state_dim), dtype=np.float32)
        self.remote_states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.true_targets = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.next_delayed_sequences = np.zeros((buffer_size, self.seq_len, self.state_dim), dtype=np.float32)
        self.next_remote_states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, delayed_seq, remote_state, action, reward, true_target, next_delayed_seq, next_remote_state, done):
        self.delayed_sequences[self.ptr] = delayed_seq
        self.remote_states[self.ptr] = remote_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.true_targets[self.ptr] = true_target
        self.next_delayed_sequences[self.ptr] = next_delayed_seq
        self.next_remote_states[self.ptr] = next_remote_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {
            'delayed_sequences': torch.tensor(self.delayed_sequences[indices], device=self.device),
            'remote_states': torch.tensor(self.remote_states[indices], device=self.device),
            'actions': torch.tensor(self.actions[indices], device=self.device),
            'rewards': torch.tensor(self.rewards[indices], device=self.device),
            'true_targets': torch.tensor(self.true_targets[indices], device=self.device),
            'next_delayed_sequences': torch.tensor(self.next_delayed_sequences[indices], device=self.device),
            'next_remote_states': torch.tensor(self.next_remote_states[indices], device=self.device),
            'dones': torch.tensor(self.dones[indices], device=self.device)
        }
        return batch


class SACTrainer:
    
    def __init__(self,
                 env: VecEnv,
                 pretrained_estimator_path: Optional[str] = None
                 ):
        """Initialize the Model-Based SAC Trainer."""

        self.env = env
        self.num_envs = env.num_envs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load LSTM State Estimator
        self.state_estimator = StateEstimator().to(self.device)

        # Auto-find best LSTM model if path not provided
        if pretrained_estimator_path is None:
            pretrained_estimator_path = self._find_best_lstm_model()

        if pretrained_estimator_path and os.path.exists(pretrained_estimator_path):
            logger.info(f"Loading pre-trained LSTM from: {pretrained_estimator_path}")
            weights = torch.load(pretrained_estimator_path, map_location=self.device)
            self.state_estimator.load_state_dict(weights['state_estimator_state_dict'])
            logger.info("Pre-trained LSTM loaded successfully")
        else:
            logger.warning(f"No pre-trained LSTM found at: {pretrained_estimator_path}")
            logger.warning("Initializing LSTM with random weights - training may be unstable!")
            logger.warning("Please run LSTM pre-training first for better results.")

        self.state_estimator.eval() # Set to evaluation mode
        for param in self.state_estimator.parameters():
            param.requires_grad = False  # Freeze parameters

        # Initialize Actor and Critic Networks
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Initialize Actor and Critic Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=SAC_LEARNING_RATE
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=SAC_LEARNING_RATE
        )
        
        # Automatic Temperature (Alpha) Tuning (unchanged)
        if SAC_TARGET_ENTROPY == 'auto':
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        else:
            self.target_entropy = float(SAC_TARGET_ENTROPY)
            
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=ALPHA_LEARNING_RATE)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(SAC_BUFFER_SIZE, self.device)

        # Training state
        self.total_timesteps = 0
        self.num_updates = 0
        self.checkpoint_dir = CHECKPOINT_DIR_RL
        self.tb_writer: Optional[SummaryWriter] = None
        self.training_start_time = None
        self.hidden_states = self.state_estimator.init_hidden_state(self.num_envs, self.device)

        # Early stopping tracking
        self.best_eval_reward = -np.inf
        self.eval_patience_counter = 0
        self.eval_history = []

    def _find_best_lstm_model(self) -> Optional[str]:
        """Auto-find the best pre-trained LSTM model."""
        import glob

        # Search in common locations
        search_paths = [
            os.path.join(CHECKPOINT_DIR_LSTM, "*/estimator_best.pth"),
            os.path.join("./lstm_training_output", "*/estimator_best.pth"),
            os.path.join(os.path.dirname(__file__), "lstm_training_output", "*/estimator_best.pth"),
        ]

        all_models = []
        for pattern in search_paths:
            all_models.extend(glob.glob(pattern))

        if not all_models:
            return None

        # Return the most recent model
        all_models.sort(key=os.path.getmtime, reverse=True)
        return all_models[0]

    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        tb_dir = os.path.join(self.checkpoint_dir, "tensorboard_sac")
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=120)
        logger.info(f"Tensorboard logs at: {tb_dir}")

    def _log_metrics(self, metrics: Dict[str, float], avg_reward: float):
        step = self.num_updates
        if self.tb_writer:
            self.tb_writer.add_scalar('train/avg_episode_reward', avg_reward, self.total_timesteps)
            self.tb_writer.add_scalar('train/alpha', metrics.get('alpha', 0.0), step)
            self.tb_writer.add_scalars('losses', {
                'actor_loss': metrics.get('actor_loss', 0.0),
                'critic_loss': metrics.get('critic_loss', 0.0),
                'alpha_loss': metrics.get('alpha_loss', 0.0),
            }, step)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().detach().item()

    def select_action(self, 
                      delayed_seq_batch: np.ndarray, 
                      remote_state_batch: np.ndarray,
                      deterministic: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            delayed_seq_t = torch.tensor(delayed_seq_batch, dtype=torch.float32, device=self.device)
            remote_state_t = torch.tensor(remote_state_batch, dtype=torch.float32, device=self.device)

            predicted_state_t, new_hidden_state = self.state_estimator(
                delayed_seq_t, self.hidden_states
            )
            self.hidden_states = new_hidden_state
            
            actor_input_t = torch.cat([predicted_state_t, remote_state_t], dim=1)
            
            action_t, _, _ = self.actor.sample(actor_input_t, deterministic)
            
            actions_np = action_t.cpu().numpy()
            predicted_states_np = predicted_state_t.cpu().numpy()

            return actions_np, predicted_states_np

    def evaluate(self, num_episodes: int = SAC_VAL_EPISODES, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the current policy over multiple episodes.

        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Starting evaluation for {num_episodes} episodes...")

        # Import here to avoid circular dependency
        from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
        from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig

        # Create a single evaluation environment (no vectorization for cleaner evaluation)
        eval_env = TeleoperationEnvWithDelay(
            delay_config=ExperimentConfig.MEDIUM_DELAY,  # Use same config as training
            trajectory_type=self.env.env_method("get_trajectory_type")[0] if hasattr(self.env, 'env_method') else None,
            randomize_trajectory=False,  # Fixed trajectory for fair comparison
            render_mode=None
        )

        episode_rewards = []
        episode_lengths = []
        episode_tracking_errors = []
        episode_prediction_errors = []

        for ep in range(num_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            tracking_errors = []
            prediction_errors = []

            # Reset hidden state for new episode
            eval_hidden_state = self.state_estimator.init_hidden_state(1, self.device)

            done = False
            while not done:
                # Get delayed sequence and remote state
                delayed_seq = eval_env.get_delayed_target_buffer(RNN_SEQUENCE_LENGTH)
                delayed_seq = delayed_seq.reshape(1, RNN_SEQUENCE_LENGTH, N_JOINTS * 2)
                remote_state = eval_env.get_remote_state().reshape(1, -1)

                # Select action
                with torch.no_grad():
                    delayed_seq_t = torch.tensor(delayed_seq, dtype=torch.float32, device=self.device)
                    remote_state_t = torch.tensor(remote_state, dtype=torch.float32, device=self.device)

                    predicted_state_t, eval_hidden_state = self.state_estimator(
                        delayed_seq_t, eval_hidden_state
                    )

                    actor_input_t = torch.cat([predicted_state_t, remote_state_t], dim=1)
                    action_t, _, _ = self.actor.sample(actor_input_t, deterministic)
                    action = action_t.cpu().numpy()[0]

                # Set predicted target before step
                eval_env.set_predicted_target(predicted_state_t.cpu().numpy()[0])

                # Take step
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                # Track errors
                if 'real_time_joint_error' in info:
                    tracking_errors.append(info['real_time_joint_error'])
                if 'prediction_error' in info and not np.isnan(info['prediction_error']):
                    prediction_errors.append(info['prediction_error'])

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if tracking_errors:
                episode_tracking_errors.append(np.mean(tracking_errors))
            if prediction_errors:
                episode_prediction_errors.append(np.mean(prediction_errors))

        eval_env.close()

        # Calculate statistics
        eval_metrics = {
            'eval/mean_reward': np.mean(episode_rewards),
            'eval/std_reward': np.std(episode_rewards),
            'eval/min_reward': np.min(episode_rewards),
            'eval/max_reward': np.max(episode_rewards),
            'eval/mean_length': np.mean(episode_lengths),
            'eval/mean_tracking_error': np.mean(episode_tracking_errors) if episode_tracking_errors else 0.0,
            'eval/mean_prediction_error': np.mean(episode_prediction_errors) if episode_prediction_errors else 0.0,
        }

        logger.info(f"Evaluation complete:")
        logger.info(f"  Mean Reward: {eval_metrics['eval/mean_reward']:.3f} ± {eval_metrics['eval/std_reward']:.3f}")
        logger.info(f"  Mean Episode Length: {eval_metrics['eval/mean_length']:.1f}")
        logger.info(f"  Mean Tracking Error: {eval_metrics['eval/mean_tracking_error']*1000:.2f}mm")
        if episode_prediction_errors:
            logger.info(f"  Mean Prediction Error: {eval_metrics['eval/mean_prediction_error']*1000:.2f}mm")

        return eval_metrics

    def update_policy(self) -> Dict[str, float]:
        """
        Perform one SAC update step (Estimator is frozen).
        """
        metrics = {}
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(SAC_BATCH_SIZE)
        
        # The estimator is frozen, so we do not update it.
        # We can, however, log its current performance.
        with torch.no_grad():
            predicted_targets, _ = self.state_estimator(batch['delayed_sequences'])
            estimator_loss = F.mse_loss(predicted_targets, batch['true_targets'])
            metrics['estimator_loss (frozen)'] = estimator_loss.item()
        
        with torch.no_grad():
            # Get predicted states (from frozen estimator)
            predicted_state_t, _ = self.state_estimator(batch['delayed_sequences'])
            next_predicted_state_t, _ = self.state_estimator(batch['next_delayed_sequences'])
            
            actor_state_t = torch.cat([predicted_state_t, batch['remote_states']], dim=1)
            next_actor_state_t = torch.cat([next_predicted_state_t, batch['next_remote_states']], dim=1)

            # Get next actions and log_probs from *current* actor
            next_actions_t, next_log_probs_t, _ = self.actor.sample(next_actor_state_t)
            
            # Get Q-values from *target* critic
            q1_target, q2_target = self.critic_target(next_actor_state_t, next_actions_t)
            q_target_min = torch.min(q1_target, q2_target)
            
            # Calculate the target for the Q-function
            q_target = batch['rewards'] + (1.0 - batch['dones']) * SAC_GAMMA * (
                q_target_min - self.alpha * next_log_probs_t
            )

        # Get current Q-values from *current* critic
        current_q1, current_q2 = self.critic(actor_state_t, batch['actions'])
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        metrics['critic_loss'] = critic_loss.item()
        
        # Update Actor Network
        for param in self.critic.parameters():
            param.requires_grad = False
            
        # Get actions and log_probs from current actor
        actions_t, log_probs_t, _ = self.actor.sample(actor_state_t)
        
        q1_policy, q2_policy = self.critic(actor_state_t, actions_t)
        q_policy_min = torch.min(q1_policy, q2_policy)
        
        # Actor loss
        actor_loss = (self.alpha * log_probs_t - q_policy_min).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        metrics['actor_loss'] = actor_loss.item()
        
        for param in self.critic.parameters():
            param.requires_grad = True
            
        # Update Alpha (Temperature)
        alpha_loss = -(self.log_alpha * (log_probs_t + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha'] = self.alpha
        
        # Update Target Networks
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1.0 - SAC_TAU)
                target_param.data.add_(SAC_TAU * param.data)
                
        return metrics
        
    def train(self, total_timesteps: int):
        """ The main training loop for Model-Based SAC. """
        self._init_tensorboard()
        self.training_start_time = datetime.now()
        start_time = self.training_start_time
        self.env.reset()
        episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        completed_episode_rewards = deque(maxlen=100)
        logger.info("Starting training...")

        for t in range(int(total_timesteps // self.num_envs)): # collecting data
            delayed_buffers_list = self.env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
            remote_states_list = self.env.env_method("get_remote_state")
            true_targets_list = self.env.env_method("get_true_current_target")
            delayed_seq_batch = np.array([buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2) for buf in delayed_buffers_list])
            remote_state_batch = np.array(remote_states_list)
            true_target_batch = np.array(true_targets_list)

            if self.total_timesteps < SAC_START_STEPS:
                actions_batch = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
                predicted_state_batch = np.zeros_like(remote_state_batch)
            else:
                actions_batch, predicted_state_batch = self.select_action(
                    delayed_seq_batch, remote_state_batch
                )

            for i in range(self.num_envs):
                self.env.env_method("set_predicted_target", predicted_state_batch[i], indices=[i])
            
            _, rewards_batch, dones_batch, infos_batch = self.env.step(actions_batch)
            
            next_delayed_buffers_list = self.env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
            next_remote_states_list = self.env.env_method("get_remote_state")
            next_delayed_seq_batch = np.array([buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2) for buf in next_delayed_buffers_list])
            next_remote_state_batch = np.array(next_remote_states_list)
            
            for i in range(self.num_envs):
                self.replay_buffer.add(
                    delayed_seq_batch[i], remote_state_batch[i], actions_batch[i],
                    rewards_batch[i], true_target_batch[i],
                    next_delayed_seq_batch[i], next_remote_state_batch[i],
                    dones_batch[i]
                )

            self.total_timesteps += self.num_envs
            episode_rewards += rewards_batch
            episode_lengths += 1

            for i in range(self.num_envs):
                if dones_batch[i]:
                    completed_episode_rewards.append(episode_rewards[i])
                    if isinstance(self.hidden_states, tuple):
                        h, c = self.hidden_states
                        h[:, i, :] = 0.0
                        c[:, i, :] = 0.0
                        self.hidden_states = (h, c)
                    else:
                        self.hidden_states[:, i, :] = 0.0
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
            
            # Update policy after collecting sufficient data
            if self.total_timesteps >= SAC_START_STEPS:
                for _ in range(int(SAC_UPDATES_PER_STEP * self.num_envs)):
                    metrics = self.update_policy()
                    self.num_updates += 1

            # Logging
            if (t + 1) % (LOG_FREQ * 10) == 0:
                elapsed_time = datetime.now() - start_time
                avg_reward = np.mean(completed_episode_rewards) if completed_episode_rewards else 0.0
                logger.info(f"\n{'─'*70}")
                logger.info(f"Timesteps: {self.total_timesteps:,} | Updates: {self.num_updates:,} | Elapsed: {str(elapsed_time).split('.')[0]}")
                logger.info(f"  Avg Reward (last 100): {avg_reward:.3f}")
                if self.total_timesteps >= SAC_START_STEPS:
                    logger.info(f"  Estimator Loss (Frozen): {metrics.get('estimator_loss (frozen)', 0):.6f}")
                    logger.info(f"  Actor Loss: {metrics.get('actor_loss', 0):.4f} | Critic Loss: {metrics.get('critic_loss', 0):.4f}")
                    logger.info(f"  Alpha: {self.alpha:.4f} | Alpha Loss: {metrics.get('alpha_loss', 0):.4f}")
                self._log_metrics(metrics, avg_reward)

            # Evaluation and Early Stopping
            if self.total_timesteps > 0 and self.total_timesteps % SAC_VAL_FREQ == 0:
                logger.info(f"\n{'='*70}")
                logger.info(f"EVALUATION at Timestep {self.total_timesteps:,}")
                logger.info(f"{'='*70}")

                # Run evaluation
                eval_metrics = self.evaluate(num_episodes=SAC_VAL_EPISODES)

                # Log to tensorboard
                if self.tb_writer:
                    for key, value in eval_metrics.items():
                        self.tb_writer.add_scalar(key, value, self.total_timesteps)

                # Track evaluation history
                self.eval_history.append({
                    'timestep': self.total_timesteps,
                    'mean_reward': eval_metrics['eval/mean_reward']
                })

                # Check for improvement
                current_eval_reward = eval_metrics['eval/mean_reward']
                if current_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = current_eval_reward
                    self.eval_patience_counter = 0

                    # Save best model
                    best_model_path = os.path.join(self.checkpoint_dir, "best_policy.pth")
                    logger.info(f"  🎉 New best evaluation reward: {self.best_eval_reward:.3f}")
                    logger.info(f"  💾 Saving best model to: {best_model_path}")
                    self.save_checkpoint("best_policy.pth")
                else:
                    self.eval_patience_counter += 1
                    logger.info(f"  ⚠️  No improvement for {self.eval_patience_counter}/{SAC_EARLY_STOPPING_PATIENCE} evaluations")
                    logger.info(f"  Best reward so far: {self.best_eval_reward:.3f}")

                    # Check early stopping
                    if self.eval_patience_counter >= SAC_EARLY_STOPPING_PATIENCE:
                        logger.info(f"\n{'='*70}")
                        logger.info("EARLY STOPPING TRIGGERED")
                        logger.info(f"No improvement for {SAC_EARLY_STOPPING_PATIENCE} consecutive evaluations")
                        logger.info(f"Best evaluation reward: {self.best_eval_reward:.3f}")
                        logger.info(f"{'='*70}\n")

                        # Save final model before stopping
                        self.save_checkpoint("early_stopped_policy.pth")
                        break

                logger.info(f"{'='*70}\n")

            # Save checkpoint
            if (t + 1) % (SAVE_FREQ * 10) == 0:
                self.save_checkpoint(f"policy_step_{self.total_timesteps}.pth")

        logger.info("Training completed. Saving final models.")
        self.save_checkpoint("final_policy.pth")
        if self.tb_writer:
            self.tb_writer.close()

    def save_checkpoint(self, filename: str):
        """Save the current model checkpoint."""
        
        path = os.path.join(self.checkpoint_dir, filename)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        torch.save({
            'state_estimator_state_dict': self.state_estimator.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
        }, path)
        logger.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found: {path}")
            return
            
        checkpoint = torch.load(path, map_location=self.device)
        
        self.state_estimator.load_state_dict(checkpoint['state_estimator_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.num_updates = checkpoint.get('num_updates', 0)
        
        # Ensure estimator is frozen after loading
        self.state_estimator.eval()
        for param in self.state_estimator.parameters():
            param.requires_grad = False
            
        logger.info(f"Checkpoint loaded from: {path}")