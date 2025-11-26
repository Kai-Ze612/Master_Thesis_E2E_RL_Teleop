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

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator, Actor, Critic
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import TrajectoryType

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
    MAX_EPISODE_STEPS,
    TARGET_DELTA_SCALE,
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
        
        self.state_dim = OBS_DIM
        self.action_dim = N_JOINTS
        self.target_state_dim = N_JOINTS * 2
        
        # Pre-allocate memory
        self.delayed_seq_dim = N_JOINTS * 2 + 1
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.delayed_seq_dim), dtype=np.float32)
        
        
        self.remote_states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)

        self.true_targets = np.zeros((buffer_size, self.target_state_dim), dtype=np.float32)
        
        self.next_delayed_sequences = np.zeros((buffer_size, self.seq_len, self.delayed_seq_dim), dtype=np.float32)
        
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

        # state_dim for Actor and Critic
        # Predicted state (from LSTM):           14D (7q + 7qd)
        # Remote robot augmented state:         15D (7q + 7qd + 1 delay)
        state_dim = OBS_DIM

        # Initialize Actor and Critic networks
        self.actor = Actor(state_dim=state_dim).to(self.device)
        self.critic = Critic(state_dim=state_dim).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)

        # Freeze target network parameters
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
                      obs: np.ndarray,                # [Added] Full 113D Env Observation
                      delayed_seq_batch: np.ndarray,  # (num_envs, seq_len, 15)
                      deterministic: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """ Select action using the current policy given full observation and delayed sequences. """
        with torch.no_grad():
            # 1. LSTM Prediction (Used for the Action output, not the Actor input)
            delayed_t = torch.tensor(delayed_seq_batch, dtype=torch.float32, device=self.device)
            scaled_residual_t, _ = self.state_estimator(delayed_t)
            predicted_residual_t = scaled_residual_t / TARGET_DELTA_SCALE
            
            # Calculate predicted state (for the 2nd half of the action)
            last_observation_t = delayed_t[:, -1, :N_JOINTS*2]
            predicted_state_t = last_observation_t + predicted_residual_t

            # 2. Actor Action (Uses the full 113D Observation)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_t, _, _ = self.actor.sample(obs_t, deterministic)

            actions_np = action_t.cpu().numpy()
            predicted_states_np = predicted_state_t.cpu().numpy()

            return actions_np, predicted_states_np
    
    def validate(self) -> float:
        """ Perform validation over several episodes and return average reward. """
        
        validation_rewards = []
        
        # Initialize validation environment
        val_obs, _ = self.val_env.reset() # SB3 DummyVecEnv might behave differently, but this is a standard Gym Env here
 
        for episode in range(SAC_VAL_EPISODES):
            episode_reward = 0.0
            val_obs, _ = self.val_env.reset()
            
            for step in range(MAX_EPISODE_STEPS):
                # Get delayed buffer for LSTM
                delayed_buffer = self.val_env.get_delayed_target_buffer(RNN_SEQUENCE_LENGTH)
                
                # Reshape for batch inference (1, Seq, Feat)
                delayed_seq_batch = delayed_buffer.reshape(1, RNN_SEQUENCE_LENGTH, N_JOINTS * 2 + 1)
                
                # Reshape obs for batch inference (1, 113)
                obs_batch = val_obs.reshape(1, -1)
                
                # [FIX] Use the shared select_action method
                # This handles all the LSTM prediction and Actor inference correctly
                actions, predicted_states = self.select_action(
                    obs_batch, delayed_seq_batch, deterministic=True
                )
                
                # Unpack single batch result
                action = actions[0]
                predicted_target = predicted_states[0]
                
                # Combine for environment step (21D: 7D Torque + 14D Prediction)
                augmented_action = np.concatenate([action, predicted_target])
                
                # Step environment
                val_obs, reward, done, truncated, info = self.val_env.step(augmented_action)
                
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
            # [Fix] Critic Target uses 'next_remote_states' which is actually 'next_obs' (113D)
            # We no longer need to concatenate predicted states for the Actor input, 
            # because the Actor now takes the raw 113D observation.
            
            next_state_t = batch['next_remote_states'] # This holds the 113D Next Observation
            
            # Get next actions from Target Actor
            next_actions_t, next_log_probs_t, _ = self.actor.sample(next_state_t)
            
            # Get Q-values from Target Critic
            q1_target, q2_target = self.critic_target(next_state_t, next_actions_t)
            q_target_min = torch.min(q1_target, q2_target)
            
            q_target = batch['rewards'] + (1.0 - batch['dones']) * SAC_GAMMA * (
                q_target_min - self.alpha * next_log_probs_t
            )

        # [Fix] Current Critic Update
        current_state_t = batch['remote_states'] # This holds the 113D Current Observation
        current_q1, current_q2 = self.critic(current_state_t, batch['actions'])
        
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update Actor Network
        for param in self.critic.parameters():
            param.requires_grad = False
            
        # Get actions and log_probs from current actor
        actions_t, log_probs_t, _ = self.actor.sample(current_state_t)
        q1_policy, q2_policy = self.critic(current_state_t, actions_t)
        q_policy_min = torch.min(q1_policy, q2_policy)
        
        # Actor loss: maximize (Q - α*log_π) = minimize (α*log_π - Q)
        actor_loss = (self.alpha * log_probs_t - q_policy_min).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # <--- ADD THIS
        self.actor_optimizer.step()
        
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
        validation_reward = None
        
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

        obs = self.env.reset()
        
        for t in range(int(total_timesteps // self.num_envs)):
            
            # Debug Print to confirm it is moving
            if t % 100 == 0:
                print(f"Step {self.total_timesteps}/{total_timesteps}", end='\r')

            # 1. Data Collection (Get History)
            delayed_buffers_list = self.env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
            true_targets_list = self.env.env_method("get_true_current_target")
            
            # Stack and Reshape: (Batch, Seq, Feat)
            flat_batch = np.stack(delayed_buffers_list)
            delayed_seq_batch = flat_batch.reshape(self.num_envs, RNN_SEQUENCE_LENGTH, N_JOINTS * 2 + 1)
            true_target_batch = np.array(true_targets_list)

            # 2. Action Selection
            policy_actions, predicted_state_batch = self.select_action(
                obs, delayed_seq_batch, deterministic=False
            )

            # 3. Exploration Logic
            if self.total_timesteps < SAC_START_STEPS:
                actions_batch = np.array([
                    np.random.uniform(-1.0, 1.0, size=(N_JOINTS,)) 
                    for _ in range(self.num_envs)
                ])
            else:
                actions_batch = policy_actions

            # 4. Step Environment
            augmented_actions_batch = np.concatenate([actions_batch, predicted_state_batch], axis=1)
            next_obs, rewards_batch, dones_batch, infos_batch = self.env.step(augmented_actions_batch)
            
            # 5. Get Next History (for Replay Buffer)
            next_delayed_buffers_list = self.env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
            next_flat_batch = np.stack(next_delayed_buffers_list)
            next_delayed_seq_batch = next_flat_batch.reshape(self.num_envs, RNN_SEQUENCE_LENGTH, N_JOINTS * 2 + 1)

            # 6. Store Experience in Replay Buffer
            # [FIX] Iterate over ENVIRONMENTS (i), not Time (t)
            for i in range(self.num_envs):
                if infos_batch[i].get('is_in_warmup', False):
                    continue
                
                self.replay_buffer.add(
                    delayed_seq_batch[i],
                    obs[i],                # 113D Current
                    actions_batch[i],      # 7D Torque
                    rewards_batch[i],
                    true_target_batch[i],
                    next_delayed_seq_batch[i],
                    next_obs[i],           # 113D Next
                    dones_batch[i]
                )

            obs = next_obs
            
            # 7. Update Statistics
            self.total_timesteps += self.num_envs
            episode_rewards += rewards_batch
            episode_lengths += 1

            # Track Episode Completion
            for i in range(self.num_envs):
                if dones_batch[i]:
                    completed_episode_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                
                # Track Metrics
                info = infos_batch[i]
                self.real_time_error_history.append(info.get('real_time_joint_error', np.nan))
                self.prediction_error_history.append(info.get('prediction_error', np.nan))
                self.delay_steps_history.append(info.get('current_delay_steps', np.nan))

            # 8. Policy Update (Gradient Descent)
            if self.total_timesteps >= SAC_START_STEPS:
                for _ in range(int(SAC_UPDATES_PER_STEP * self.num_envs)):
                    metrics = self.update_policy()
                    self.num_updates += 1

            # 9. Validation & Logging
            if (t + 1) % (SAC_VAL_FREQ // self.num_envs) == 0 and self.total_timesteps >= SAC_START_STEPS:
                validation_reward = self.validate()
                self.validation_rewards_history.append(validation_reward)
                
                if validation_reward > self.best_validation_reward:
                    self.best_validation_reward = validation_reward
                    self.patience_counter = 0
                    self.save_checkpoint("best_policy.pth")
                    logger.info(f"✓ NEW BEST! Val Reward: {validation_reward:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= SAC_EARLY_STOPPING_PATIENCE:
                        logger.info("Early stopping triggered.")
                        self.early_stop_triggered = True
                        break

            # Periodic Console Log
            if (t + 1) % (LOG_FREQ * 10) == 0:
                elapsed_time = datetime.now() - start_time
                avg_reward = np.mean(completed_episode_rewards) if completed_episode_rewards else 0.0
                
                logger.info(f"\n{'─'*70}")
                logger.info(f"Timesteps: {self.total_timesteps:,} | Updates: {self.num_updates:,}")
                logger.info(f"Avg Reward: {avg_reward:.4f}")
                
                env_stats = {
                    'avg_rt_error_rad': np.nanmean(self.real_time_error_history),
                    'avg_pred_error_rad': np.nanmean(self.prediction_error_history),
                    'avg_delay': np.nanmean(self.delay_steps_history)
                }
                self._log_metrics(metrics, avg_reward, validation_reward, env_stats)

        # Save Final
        self.save_checkpoint("final_policy.pth")
        if self.tb_writer: self.tb_writer.close()

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