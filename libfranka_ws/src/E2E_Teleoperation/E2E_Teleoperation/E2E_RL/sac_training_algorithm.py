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

from E2E_Teleoperation.E2E_RL.sac_policy_network import JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnvWithDelay
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig
from E2E_Teleoperation.E2E_RL.local_robot_simulator import TrajectoryType
import E2E_Teleoperation.config.robot_config as cfg

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """
    Store all collected experience for SAC training.
    [MODIFICATION] Added storage for 'true_states' (Ground Truth).
    """
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state_dim = cfg.OBS_DIM
        self.action_dim = cfg.N_JOINTS # 7D (Torque only)
        # Ground truth state dim is 14 (7q + 7qd)
        self.true_state_dim = 14 
        
        self.remote_states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_remote_states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        
        # [MODIFICATION] Store ground truth for Aux Loss
        self.true_states = np.zeros((buffer_size, self.true_state_dim), dtype=np.float32)
    
    def add(self, remote_state, action, reward, next_remote_state, done, true_state):
        """Add a single experience to the replay buffer"""
        self.remote_states[self.ptr] = remote_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_remote_states[self.ptr] = next_remote_state
        self.dones[self.ptr] = done
        self.true_states[self.ptr] = true_state 
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences from the replay buffer"""
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {
            'remote_states': torch.tensor(self.remote_states[indices], device=self.device),
            'actions': torch.tensor(self.actions[indices], device=self.device),
            'rewards': torch.tensor(self.rewards[indices], device=self.device),
            'next_remote_states': torch.tensor(self.next_remote_states[indices], device=self.device),
            'dones': torch.tensor(self.dones[indices], device=self.device),
            'true_states': torch.tensor(self.true_states[indices], device=self.device)
        }
        return batch
    
    def __len__(self) -> int:
        return self.size


class SACTrainer:
    """Soft actor-critic (SAC) training algorithm with Joint Training (Aux Loss)."""
    
    def __init__(self, 
                 env: VecEnv,
                 val_delay_config: ExperimentConfig = ExperimentConfig.LOW_DELAY
                 ):
        
        self.env = env
        self.num_envs = env.num_envs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Validation Environment
        # Removed lstm_model_path (Env handles raw history now)
        self.val_env = TeleoperationEnvWithDelay(
            delay_config=val_delay_config,
            trajectory_type=TrajectoryType.FIGURE_8,
            randomize_trajectory=False,
            render_mode=None
        )
        
        # Initialize Joint Networks
        # [MODIFICATION] Use cfg.* for dimensions
        self.actor = JointActor(action_dim=cfg.N_JOINTS).to(self.device)
        self.critic = JointCritic(action_dim=cfg.N_JOINTS).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)

        # Freeze target network parameters
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Initialize Optimizers
        # [MODIFICATION] Use cfg.* for learning rates
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.SAC_LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.SAC_LEARNING_RATE)

        # Automatic Entropy Tuning
        if cfg.SAC_TARGET_ENTROPY == 'auto':
            self.target_entropy = -float(cfg.N_JOINTS)
        else:
            self.target_entropy = float(cfg.SAC_TARGET_ENTROPY)
            
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.ALPHA_LEARNING_RATE)
        
        # Replay Buffer
        # [MODIFICATION] Use cfg.SAC_BUFFER_SIZE
        self.replay_buffer = ReplayBuffer(cfg.SAC_BUFFER_SIZE, self.device)

        # Training state
        self.total_timesteps = 0
        self.num_updates = 0
        self.checkpoint_dir = cfg.CHECKPOINT_DIR_RL
        self.tb_writer: Optional[SummaryWriter] = None
        self.training_start_time = None

        # Validation stats
        self.best_validation_reward = -np.inf
        self.validation_rewards_history = deque(maxlen=100)
        self.patience_counter = 0

        self.real_time_error_history = deque(maxlen=1000)
        self.prediction_error_history = deque(maxlen=1000)
        self.delay_steps_history = deque(maxlen=1000)
        
    def _init_tensorboard(self):
        tb_dir = os.path.join(self.checkpoint_dir, "tensorboard_sac")
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=120)
        logger.info(f"Tensorboard logs at: {tb_dir}")

    def _log_metrics(self, metrics: Dict[str, float], avg_reward: float, val_reward: Optional[float] = None,
                     env_stats: Dict[str, float] = {}):
        step = self.num_updates
        if self.tb_writer:
            self.tb_writer.add_scalar('train/avg_reward', avg_reward, self.total_timesteps)
            if val_reward is not None:
                self.tb_writer.add_scalar('train/val_reward', val_reward, self.total_timesteps)

            if env_stats:
                self.tb_writer.add_scalar('env/real_time_error_q_rad', env_stats.get('avg_rt_error_rad', np.nan), self.total_timesteps)
                self.tb_writer.add_scalar('env/prediction_error_lstm_rad', env_stats.get('avg_pred_error_rad', np.nan), self.total_timesteps)
                self.tb_writer.add_scalar('env/avg_delay_steps', env_stats.get('avg_delay', np.nan), self.total_timesteps)

            self.tb_writer.add_scalar('train/alpha', metrics.get('alpha', 0.0), step)
            self.tb_writer.add_scalars('losses', {
                'actor_loss': metrics.get('actor_loss', 0.0),
                'critic_loss': metrics.get('critic_loss', 0.0),
                'alpha_loss': metrics.get('alpha_loss', 0.0),
                'aux_loss': metrics.get('aux_loss', 0.0)
            }, step)
    
    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().detach().item()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """ 
        Select action (Torque) using the current policy given full observation.
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            # JointActor.sample returns: scaled_action, log_prob, raw_action, pred_state
            action_t, _, _, _ = self.actor.sample(obs_t, deterministic)
            return action_t.cpu().numpy()
    
    def validate(self) -> float:
        """ Perform validation over several episodes. """
        validation_rewards = []
 
        for episode in range(cfg.SAC_VAL_EPISODES):
            episode_reward = 0.0
            val_obs, _ = self.val_env.reset()
            
            for step in range(cfg.MAX_EPISODE_STEPS):
                obs_batch = val_obs.reshape(1, -1)
                
                # Get Action (Torque)
                actions = self.select_action(obs_batch, deterministic=True)
                action = actions[0]
                
                val_obs, reward, done, truncated, info = self.val_env.step(action)
                episode_reward += reward
                
                if done or truncated:
                    break
            
            validation_rewards.append(episode_reward)
        
        avg_reward = np.mean(validation_rewards)
        logger.info(f"Validation Reward: {avg_reward:.4f}")
        return avg_reward
            
    def update_policy(self) -> Dict[str, float]:
        """ Perform one Joint SAC update step. """
        metrics = {}
        # [MODIFICATION] Use cfg.SAC_BATCH_SIZE
        if len(self.replay_buffer) < cfg.SAC_BATCH_SIZE:
            return metrics
        
        batch = self.replay_buffer.sample(cfg.SAC_BATCH_SIZE)
        
        # --- 1. Critic Update (Standard) ---
        with torch.no_grad():
            next_state_t = batch['next_remote_states'] 
            # Note: We ignore pred_state (_) here for the target calculation
            next_actions_t, next_log_probs_t, _, _ = self.actor.sample(next_state_t)
            
            q1_target, q2_target = self.critic_target(next_state_t, next_actions_t)
            q_target_min = torch.min(q1_target, q2_target)
            
            # [MODIFICATION] Use cfg.SAC_GAMMA
            q_target = batch['rewards'] + (1.0 - batch['dones']) * cfg.SAC_GAMMA * (q_target_min - self.alpha * next_log_probs_t)

        current_state_t = batch['remote_states']
        current_q1, current_q2 = self.critic(current_state_t, batch['actions'])
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # --- 2. Actor Update (Joint) ---
        # Freeze critic for efficiency
        for param in self.critic.parameters(): param.requires_grad = False
            
        actions_t, log_probs_t, _, pred_state = self.actor.sample(current_state_t)
        
        q1_policy, q2_policy = self.critic(current_state_t, actions_t)
        q_policy_min = torch.min(q1_policy, q2_policy)
        
        # RL Loss component
        rl_loss = (self.alpha * log_probs_t - q_policy_min).mean()
        
        # [MODIFICATION] Auxiliary Loss component (MSE vs Ground Truth)
        aux_loss = F.mse_loss(pred_state, batch['true_states'])
        
        # Combined Loss (Weight = 1.0 for now)
        total_actor_loss = rl_loss + (1.0 * aux_loss)
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Unfreeze critic
        for param in self.critic.parameters(): param.requires_grad = True
            
        # --- 3. Alpha Update ---
        alpha_loss = -(self.log_alpha * (log_probs_t + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        metrics.update({
            'actor_loss': rl_loss.item(),
            'aux_loss': aux_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha
        })
        
        # --- 4. Soft Update Target Networks ---
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                # [MODIFICATION] Use cfg.SAC_TAU
                target_param.data.mul_(1.0 - cfg.SAC_TAU)
                target_param.data.add_(cfg.SAC_TAU * param.data)
                
        return metrics
       
    def train(self, total_timesteps: int):
        self._init_tensorboard()
        self.training_start_time = datetime.now()
        
        metrics = {}
        
        logger.info("="*70)
        logger.info("Starting Joint SAC Training (Recurrent E2E)")
        # [MODIFICATION] Use cfg.OBS_DIM, cfg.N_JOINTS
        logger.info(f"  Obs Dim: {cfg.OBS_DIM} | Output Dim: {cfg.N_JOINTS}")
        logger.info("="*70)
        
        episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        completed_episode_rewards = deque(maxlen=100)
        
        obs = self.env.reset()
        
        for t in range(int(total_timesteps // self.num_envs)):
            
            # 1. Action Selection
            # [MODIFICATION] Use cfg.SAC_START_STEPS
            if self.total_timesteps < cfg.SAC_START_STEPS:
                actions_batch = np.array([
                    # [MODIFICATION] Use cfg.N_JOINTS
                    np.random.uniform(-1.0, 1.0, size=(cfg.N_JOINTS,)) 
                    for _ in range(self.num_envs)
                ])
                # Scale random actions to torque limits
                # Note: 87.0 is hardcoded as approximate limit, ideally fetch from cfg if available as simple array
                # For safety, using explicit numbers matching config
                actions_batch = actions_batch * np.array([87.0]*4 + [12.0]*3) 
            else:
                actions_batch = self.select_action(obs)

            # 2. Step Environment
            next_obs, rewards_batch, dones_batch, infos_batch = self.env.step(actions_batch)
            
            # 3. Store Experience
            for i in range(self.num_envs):
                if infos_batch[i].get('is_in_warmup', False):
                    continue
                
                # Extract True State from Info
                true_state = infos_batch[i].get('true_state', np.zeros(14, dtype=np.float32))

                self.replay_buffer.add(
                    obs[i],
                    actions_batch[i],
                    rewards_batch[i],
                    next_obs[i],
                    dones_batch[i],
                    true_state
                )

            obs = next_obs
            
            # 4. Stats
            self.total_timesteps += self.num_envs
            episode_rewards += rewards_batch

            for i in range(self.num_envs):
                if dones_batch[i]:
                    completed_episode_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0
                
                info = infos_batch[i]
                self.real_time_error_history.append(info.get('real_time_joint_error', np.nan))
                self.prediction_error_history.append(info.get('prediction_error', np.nan))
                self.delay_steps_history.append(info.get('current_delay_steps', np.nan))

            # 5. Policy Update
            if self.total_timesteps >= cfg.SAC_START_STEPS:
                for _ in range(int(cfg.SAC_UPDATES_PER_STEP * self.num_envs)):
                    metrics = self.update_policy()
                    self.num_updates += 1

            # 6. Validation
            # [MODIFICATION] Use cfg constants
            if (t + 1) % (cfg.SAC_VAL_FREQ // self.num_envs) == 0 and self.total_timesteps >= cfg.SAC_START_STEPS:
                val_reward = self.validate()
                self.validation_rewards_history.append(val_reward)
                if val_reward > self.best_validation_reward:
                    self.best_validation_reward = val_reward
                    self.patience_counter = 0
                    self.save_checkpoint("best_policy.pth")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= cfg.SAC_EARLY_STOPPING_PATIENCE:
                        logger.info("Early stopping.")
                        break

            # 7. Logging
            if (t + 1) % (cfg.LOG_FREQ * 10) == 0:
                avg_reward = np.mean(completed_episode_rewards) if completed_episode_rewards else 0.0
                env_stats = {
                    'avg_rt_error_rad': np.nanmean(self.real_time_error_history),
                    'avg_pred_error_rad': np.nanmean(self.prediction_error_history),
                    'avg_delay': np.nanmean(self.delay_steps_history)
                }
                self._log_metrics(metrics, avg_reward, None, env_stats)

        self.save_checkpoint("final_policy.pth")
        if self.tb_writer: self.tb_writer.close()

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'log_alpha': self.log_alpha,
            'total_timesteps': self.total_timesteps,
            'best_validation_reward': self.best_validation_reward,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path): return
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.log_alpha = checkpoint['log_alpha']