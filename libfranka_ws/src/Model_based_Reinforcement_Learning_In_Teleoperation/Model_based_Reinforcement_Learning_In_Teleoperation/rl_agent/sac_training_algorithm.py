"""
Custom Model-Based SAC (Soft Actor-Critic) Training Algorithm.

Pipeline:
1. Load pre-trained LSTM State Estimator.
2. Freeze the State Estimator parameters.
3. Data collection using the current policy.
4. Train Actor and Critic networks using SAC.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from collections import deque

from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator, Actor, Critic
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType

from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
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
    LOG_FREQ, 
    CHECKPOINT_DIR_RL,
    SAC_VAL_FREQ,
    SAC_VAL_EPISODES,
    SAC_EARLY_STOPPING_PATIENCE,
    OBS_DIM,
    MAX_EPISODE_STEPS
)

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """
    Store all collected experience for SAC training.
    
    Pipeline:
    1. Add experience
    2. Sample mini-batches for training
    """
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.seq_len = RNN_SEQUENCE_LENGTH
        
        self.state_dim = N_JOINTS * 2 + 1
        self.action_dim = N_JOINTS
        self.target_state_dim = N_JOINTS * 2
        
       
        # Pre-allocate memory
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.state_dim), dtype=np.float32)
        self.remote_states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)

        self.true_targets = np.zeros((buffer_size, self.target_state_dim), dtype=np.float32)
        
        self.next_delayed_sequences = np.zeros((buffer_size, self.seq_len, self.state_dim), dtype=np.float32)
        self.next_remote_states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, delayed_seq, remote_state, action, reward, true_target, next_delayed_seq, next_remote_state, done):
        """Add a single experience to the replay buffer"""
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
        """Sample a batch of experiences from the replay buffer"""
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
    
    def __len__(self) -> int:
        return self.size


class SACTrainer:
    """Soft actor-critic (SAC) training algorithm with pre-trained LSTM state estimator."""
    
    def __init__(self, 
                 env: VecEnv,
                 pretrained_estimator_path: Optional[str] = None,
                 val_delay_config: ExperimentConfig = ExperimentConfig.LOW_DELAY
                 ):
        
        """Initialize the Model-Based SAC Trainer."""
        
        self.env = env
        self.num_envs = env.num_envs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # create validation environment
        self.val_env = TeleoperationEnvWithDelay(
            delay_config=val_delay_config,
            trajectory_type=TrajectoryType.FIGURE_8,
            randomize_trajectory=False,  # NO randomization for validation!
            render_mode=None
        )
        
        # Load LSTM State Estimator
        self.state_estimator = StateEstimator().to(self.device)
        if pretrained_estimator_path and os.path.exists(pretrained_estimator_path):
            weights = torch.load(pretrained_estimator_path, map_location=self.device)
            self.state_estimator.load_state_dict(weights['state_estimator_state_dict'])
            logger.info(f"Loaded estimator from {pretrained_estimator_path}")
        else:
            logger.warning("Pretrained estimator path invalid or not provided. Using random weights.")

        # freeze the estimator
        self.state_estimator.eval()
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
        
        # Automatic Temperature (Alpha) Tuning
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

        # Validation parameters and Early Stopping
        self.best_validation_reward = -np.inf
        self.validation_rewards_history = deque(maxlen=100)
        self.patience_counter = 0
        self.early_stop_triggered = False

        self.real_time_error_history = deque(maxlen=1000)
        self.prediction_error_history = deque(maxlen=1000)
        self.delay_steps_history = deque(maxlen=1000)
        
    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        tb_dir = os.path.join(self.checkpoint_dir, "tensorboard_sac")
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=120)
        logger.info(f"Tensorboard logs at: {tb_dir}")

    def _log_metrics(self, metrics: Dict[str, float], avg_reward: float, val_reward: Optional[float] = None,
                     env_stats: Dict[str, float] = {}):
        """Log metrics to TensorBoard"""
        step = self.num_updates
        if self.tb_writer:
            self.tb_writer.add_scalar('train/avg_reward', avg_reward, self.total_timesteps)
            if val_reward is not None:
                self.tb_writer.add_scalar('train/val_reward', val_reward, self.total_timesteps)

            # Log detailed env stats to tensorboard in RADIANS
            if env_stats:
                self.tb_writer.add_scalar('env/real_time_error_q_rad', env_stats.get('avg_rt_error_rad', np.nan), self.total_timesteps)
                self.tb_writer.add_scalar('env/prediction_error_lstm_rad', env_stats.get('avg_pred_error_rad', np.nan), self.total_timesteps)
                self.tb_writer.add_scalar('env/avg_delay_steps', env_stats.get('avg_delay', np.nan), self.total_timesteps)

            self.tb_writer.add_scalar('train/alpha', metrics.get('alpha', 0.0), step)
            self.tb_writer.add_scalars('losses', {
                'actor_loss': metrics.get('actor_loss', 0.0),
                'critic_loss': metrics.get('critic_loss', 0.0),
                'alpha_loss': metrics.get('alpha_loss', 0.0),
            }, step)
            self.tb_writer.add_scalar('train/estimator_loss_frozen', metrics.get('estimator_loss (frozen)', 0.0), step)
    
    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().detach().item()

    def select_action(self,
                      delayed_seq_batch: np.ndarray,  # (num_envs, seq_len, 14)
                      remote_state_batch: np.ndarray,  # (num_envs, 14)
                      deterministic: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """ Select action using the current policy given delayed sequences and remote states. """
        with torch.no_grad():
            delayed_t = torch.tensor(delayed_seq_batch, dtype=torch.float32, device=self.device)  # (num_envs, seq_len, 14)
            remote_state_t = torch.tensor(remote_state_batch, dtype=torch.float32, device=self.device)  # (num_envs, 14)

            # Full-sequence LSTM (zero hidden init inside StateEstimator)
            predicted_state_t, _ = self.state_estimator(delayed_t)  # (num_envs, 14)

            actor_input_t = torch.cat([predicted_state_t, remote_state_t], dim=1)  # (num_envs, 28)

            action_t, _, _ = self.actor.sample(actor_input_t, deterministic)

            actions_np = action_t.cpu().numpy()
            predicted_states_np = predicted_state_t.cpu().numpy()

            return actions_np, predicted_states_np
    
    def validate(self) -> float:
        """ Perform validation over several episodes and return average reward. """
        
        validation_rewards = []
        
        # Initialize validation environment
        val_obs, _ = self.val_env.reset()
 
        for episode in range(SAC_VAL_EPISODES):
            episode_reward = 0.0
            val_obs, _ = self.val_env.reset()
            
            for step in range(MAX_EPISODE_STEPS):
                # Get delayed buffer for LSTM
                delayed_buffer = self.val_env.get_delayed_target_buffer(RNN_SEQUENCE_LENGTH)
                remote_state = self.val_env.get_remote_state()
                
                # Reshape for batch inference
                delayed_seq_batch = delayed_buffer.reshape(1, RNN_SEQUENCE_LENGTH, N_JOINTS * 2 + 1)
                remote_state_batch = remote_state.reshape(1, N_JOINTS * 2)
                
                # Get deterministic action (no exploration)
                with torch.no_grad():
                    delayed_t = torch.tensor(delayed_seq_batch, dtype=torch.float32, device=self.device)  # (1, seq_len, 14)
                    remote_state_t = torch.tensor(remote_state_batch, dtype=torch.float32, device=self.device)  # (1, 14)

                    # Full-sequence inference
                    predicted_state_t, _ = self.state_estimator(delayed_t)  # (1, 14)

                    actor_input_t = torch.cat([predicted_state_t, remote_state_t], dim=1)  # (1, 28)

                    action_t, _, _ = self.actor.sample(actor_input_t, deterministic=True)
                    action = action_t.cpu().numpy()[0]
                
                # Set predicted target and step environment
                self.val_env.set_predicted_target(predicted_state_t.cpu().numpy()[0])
                val_obs, reward, done, truncated, info = self.val_env.step(action)
                
                episode_reward += reward
                
                if done or truncated:
                    break
            
            validation_rewards.append(episode_reward)
            logger.info(f"  Episode {episode + 1}/{SAC_VAL_EPISODES}: Reward = {episode_reward:.4f}")
        
        avg_validation_reward = np.mean(validation_rewards)
        std_validation_reward = np.std(validation_rewards)
        
        logger.info(f"\nValidation Results:")
        logger.info(f"  Average Reward: {avg_validation_reward:.4f}")
        logger.info(f"  Std Dev: {std_validation_reward:.4f}")
        logger.info(f"  Min: {np.min(validation_rewards):.4f}")
        logger.info(f"  Max: {np.max(validation_rewards):.4f}")
        
        return avg_validation_reward
            
    def update_policy(self) -> Dict[str, float]:
        """
        Perform one SAC update step (Estimator is frozen).
        
        Pipeline:
        1. critic network update (Q learning)
        2. actor network update (policy learning)
        3. temperature parameter (entropy tuning)
        4. target network update
        """
        metrics = {}
        
        if len(self.replay_buffer) < SAC_BATCH_SIZE:
            return metrics
        
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
        
        # MSE loss between current Q and target Q (bellman equation)
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
        
        # Get Q-values from sampled actions
        q1_policy, q2_policy = self.critic(actor_state_t, actions_t)
        q_policy_min = torch.min(q1_policy, q2_policy)
        
        # Actor loss: maximize (Q - α*log_π) = minimize (α*log_π - Q)
        actor_loss = (self.alpha * log_probs_t - q_policy_min).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        metrics['actor_loss'] = actor_loss.item()
        
        # Unfreeze critic parameters
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

        metrics = {}
        
        logger.info("="*70)
        logger.info("Starting SAC Training")
        logger.info("="*70)
        logger.info(f"Configuration:")
        logger.info(f"  Environments: {self.num_envs}")
        logger.info(f"  Total timesteps: {total_timesteps:,}")
        logger.info(f"  Learning rate (Actor/Critic): {SAC_LEARNING_RATE}")
        logger.info(f"  Gamma (discount): {SAC_GAMMA}")
        logger.info(f"  Tau (soft update): {SAC_TAU}")
        logger.info(f"  Batch size: {SAC_BATCH_SIZE}")
        logger.info(f"  Buffer size: {SAC_BUFFER_SIZE:,}")
        logger.info(f"  Start steps (random): {SAC_START_STEPS:,}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Observation dim: {OBS_DIM}D")
        logger.info(f"  Actor input dim: 28D (predicted state 14D + remote state 14D)")
        logger.info("")
        
        # Initialize environment
        self.env.reset()
        episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        completed_episode_rewards = deque(maxlen=100)
        
        logger.info("Training loop starting...")
        logger.info("")

        for t in range(int(total_timesteps // self.num_envs)):
            # --- Data Collection Phase
            # Get delayed observation sequences from all environments
            delayed_buffers_list = self.env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
            remote_states_list = self.env.env_method("get_remote_state")
            true_targets_list = self.env.env_method("get_true_current_target")
            
            # Convert to numpy arrays with correct shapes
            delayed_seq_batch = np.array([
                buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2 + 1) 
                for buf in delayed_buffers_list
            ])
            remote_state_batch = np.array(remote_states_list)
            true_target_batch = np.array(true_targets_list)

            # --- Action selection phase
            policy_actions, predicted_state_batch = self.select_action(
                delayed_seq_batch, remote_state_batch, deterministic=False
            )

            # 2. Decide on the Action (Exploration vs Exploitation)
            if self.total_timesteps < SAC_START_STEPS:
                # Exploration phase: Override policy action with random action
                actions_batch = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
            else:
                # Exploitation phase: Use the policy action calculated above
                actions_batch = policy_actions

            # Set predicted targets in environments (for reward calculation)
            augmented_actions_batch = np.concatenate([actions_batch, predicted_state_batch], axis=1)
           
            # --- Environment Step
            _, rewards_batch, dones_batch, infos_batch = self.env.step(augmented_actions_batch)
            
            # Get next state observations
            next_delayed_buffers_list = self.env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
            next_remote_states_list = self.env.env_method("get_remote_state")
            next_delayed_seq_batch = np.array([buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2 + 1) for buf in next_delayed_buffers_list])
            next_remote_state_batch = np.array(next_remote_states_list)

            # --- Store Experience in Replay Buffer
            for i in range(self.num_envs):
                # [CRITICAL FIX] 1. Logic: Skip transitions collected during warmup
                # The environment forced the action to zero, but 'actions_batch' contains 
                # the policy's output. Saving this would poison the Critic.
                if infos_batch[i].get('is_in_warmup', False):
                    continue

                # [Recommended] 2. Safety: Skip if prediction contained NaNs
                # predicted_state_batch is a numpy array here
                if np.isnan(predicted_state_batch[i]).any():
                    continue
                
                info = infos_batch[i]
                current_delay_steps = info.get('current_delay_steps', 0)
                
                normalized_delay = np.array([float(current_delay_steps) / MAX_EPISODE_STEPS], dtype=np.float32)
                
                augmented_remote_state = np.concatenate([remote_state_batch[i], normalized_delay])
                augmented_next_remote_state = np.concatenate([next_remote_state_batch[i], normalized_delay])
                action_7D_torque = augmented_actions_batch[i][:N_JOINTS] # Slice the first 7 dimensions (Torque)
                
                
                # Adding back to replay buffer
                self.replay_buffer.add(
                    delayed_seq_batch[i],          # Already numpy
                    augmented_remote_state,        # Already numpy
                    action_7D_torque,              # This is the policy action (valid now because we skipped warmup)
                    rewards_batch[i],
                    true_target_batch[i],          # Already numpy
                    next_delayed_seq_batch[i],     # Already numpy
                    augmented_next_remote_state,   # Already numpy
                    dones_batch[i]
                )

            # --- Update Training Statistics
            self.total_timesteps += self.num_envs
            episode_rewards += rewards_batch
            episode_lengths += 1

            for info in infos_batch:
                self.real_time_error_history.append(info.get('real_time_joint_error', np.nan))
                self.prediction_error_history.append(info.get('prediction_error', np.nan))
                self.delay_steps_history.append(info.get('current_delay_steps', np.nan))

            # Handle episode termination
            for i in range(self.num_envs):
                if dones_batch[i]:
                    # Track completed episode reward
                    completed_episode_rewards.append(episode_rewards[i])
                    
                    # Reset episode tracking
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
            
            # --- Policy Update Phase (after collecting sufficient data)
            if self.total_timesteps >= SAC_START_STEPS:
                for _ in range(int(SAC_UPDATES_PER_STEP * self.num_envs)):
                    metrics = self.update_policy()
                    self.num_updates += 1

            # --- Validation and Early Stopping Check
            validation_reward = None
            if (t + 1) % (SAC_VAL_FREQ // self.num_envs) == 0 and self.total_timesteps >= SAC_START_STEPS:
                validation_reward = self.validate()
                self.validation_rewards_history.append(validation_reward)
                
                # Early stopping logic
                if validation_reward > self.best_validation_reward:
                    # New best!
                    self.best_validation_reward = validation_reward
                    self.patience_counter = 0
                    self.save_checkpoint("best_policy.pth")
                    logger.info(f"✓ NEW BEST! Validation reward: {validation_reward:.4f}")
                    logger.info(f"  Saved: best_policy.pth")
                else:
                    # No improvement
                    self.patience_counter += 1
                    logger.info(f"⚠ No improvement for {self.patience_counter}/{SAC_EARLY_STOPPING_PATIENCE} checks")
                    
                    # Check early stopping
                    if self.patience_counter >= SAC_EARLY_STOPPING_PATIENCE:
                        logger.info("")
                        logger.info("="*70)
                        logger.info(" EARLY STOPPING TRIGGERED!")
                        logger.info("="*70)
                        logger.info(f"Best validation reward: {self.best_validation_reward:.4f}")
                        logger.info(f"Stopped at timestep: {self.total_timesteps:,}")
                        logger.info(f"Using best checkpoint: best_policy.pth")
                        self.early_stop_triggered = True
                        break

            # Logging
            if (t + 1) % (LOG_FREQ * 10) == 0:
                elapsed_time = datetime.now() - start_time
                avg_reward = np.mean(completed_episode_rewards) if completed_episode_rewards else 0.0
                
                # Calculate info stats in RADIANS
                avg_rt_error_rad = (np.nanmean(self.real_time_error_history) 
                                    if len(self.real_time_error_history) > 0 else np.nan)
                avg_pred_error_rad = (np.nanmean(self.prediction_error_history) 
                                      if len(self.prediction_error_history) > 0 else np.nan)
                avg_delay = (np.nanmean(self.delay_steps_history) 
                             if len(self.delay_steps_history) > 0 else np.nan)
                
                logger.info(f"\n{'─'*70}")
                logger.info(f"Timesteps: {self.total_timesteps:,} | Updates: {self.num_updates:,}")
                logger.info(f"Elapsed Time: {str(elapsed_time).split('.')[0]}")
                logger.info(f"Avg Reward (last 100 ep): {avg_reward:.4f}")
                
                # Print detailed env stats in RADIANS
                logger.info(f"\nEnvironment Stats (avg over last {len(self.real_time_error_history)} steps):")
                logger.info(f" Avg Real-Time Error (q): {avg_rt_error_rad:.4f} rad")
                logger.info(f" Avg Prediction Error (LSTM): {avg_pred_error_rad:.4f} rad")
                logger.info(f" Avg Delay (steps): {avg_delay:.2f}")
                
                # Pass new stats to _log_metrics
                env_stats_dict = {
                    'avg_rt_error_rad': avg_rt_error_rad,
                    'avg_pred_error_rad': avg_pred_error_rad,
                    'avg_delay': avg_delay,
                }
                self._log_metrics(metrics, avg_reward, validation_reward, env_stats=env_stats_dict)
       
        # --- Training Complete
        logger.info("")
        logger.info("="*70)
        logger.info("Training Completed")
        logger.info("="*70)
        logger.info(f"Total timesteps: {self.total_timesteps:,}")
        logger.info(f"Total updates: {self.num_updates:,}")
        elapsed_time = datetime.now() - self.training_start_time
        logger.info(f"Total training time: {str(elapsed_time).split('.')[0]}")
        logger.info(f"Best validation reward: {self.best_validation_reward:.4f}")
        logger.info(f"Early stopping triggered: {self.early_stop_triggered}")
        logger.info("")
        logger.info("Checkpoints saved:")
        logger.info(f"  best_policy.pth (validation peak)")
        logger.info(f"  final_policy.pth (last state)")
        logger.info("")
        
        self.save_checkpoint("final_policy.pth")
        
        if self.tb_writer:
            self.tb_writer.close()

    def save_checkpoint(self, filename: str):
        """Save current model checkpoint"""
        
        path = os.path.join(self.checkpoint_dir, filename)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
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
            'best_validation_reward': self.best_validation_reward,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        
        if not os.path.exists(path):
            logger.error(f"Checkpoint not found: {path}")
            return
        
        try:
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
            self.best_validation_reward = checkpoint.get('best_validation_reward', -np.inf)
            
            # Ensure LSTM stays frozen after loading
            self.state_estimator.eval()
            for param in self.state_estimator.parameters():
                param.requires_grad = False
            
            logger.info(f"Checkpoint loaded: {path}")
            logger.info(f"  Total timesteps: {self.total_timesteps:,}")
            logger.info(f"  Total updates: {self.num_updates:,}")
            logger.info(f"  Best validation reward: {self.best_validation_reward:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise