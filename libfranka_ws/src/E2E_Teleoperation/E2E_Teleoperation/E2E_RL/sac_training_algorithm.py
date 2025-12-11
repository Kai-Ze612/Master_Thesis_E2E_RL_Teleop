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
from typing import Dict, Optional
from copy import deepcopy
from collections import deque

from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

from E2E_Teleoperation.E2E_RL.sac_policy_network import SharedLSTMEncoder, JointActor, JointCritic, create_actor_critic
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnvWithDelay
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig
from E2E_Teleoperation.E2E_RL.local_robot_simulator import TrajectoryType
import E2E_Teleoperation.config.robot_config as cfg

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Store all collected experience for SAC training."""
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state_dim = cfg.OBS_DIM
        self.action_dim = cfg.N_JOINTS
        self.true_state_dim = cfg.ESTIMATOR_OUTPUT_DIM 
        
        self.remote_states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_remote_states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self.true_states = np.zeros((buffer_size, self.true_state_dim), dtype=np.float32)
    
    def add(self, remote_state, action, reward, next_remote_state, done, true_state):
        self.remote_states[self.ptr] = remote_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_remote_states[self.ptr] = next_remote_state
        self.dones[self.ptr] = done
        self.true_states[self.ptr] = true_state 
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
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
    """
    SAC with Shared LSTM Encoder - Fixed Version.
    
    Stage 1: Train encoder AND critic (critic pre-training)
    Stage 2: Freeze encoder, train policy with delayed updates
    """
    
    def __init__(self, 
                 env: VecEnv,
                 val_delay_config: ExperimentConfig = ExperimentConfig.LOW_DELAY,
                 policy_delay: int = 2,  # TD3-style delayed updates
                 initial_alpha: float = 1.0  # Lower initial entropy
                 ):
        
        self.env = env
        self.num_envs = env.num_envs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_delay = policy_delay
        
        # Validation Environment
        self.val_env = TeleoperationEnvWithDelay(
            delay_config=val_delay_config,
            trajectory_type=TrajectoryType.FIGURE_8,
            randomize_trajectory=False,
            render_mode=None
        )
        
        # Create networks with SHARED encoder
        self.shared_encoder, self.actor, self.critic = create_actor_critic(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Parameter groups
        self.encoder_params = list(self.shared_encoder.parameters())
        self.policy_params = list(self.actor.backbone.parameters()) + \
                            list(self.actor.fc_mean.parameters()) + \
                            list(self.actor.fc_log_std.parameters())
        self.critic_q_params = list(self.critic.q1_net.parameters()) + \
                               list(self.critic.q2_net.parameters())
        
        # Separate optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder_params, lr=cfg.SAC_LEARNING_RATE)
        self.policy_optimizer = torch.optim.Adam(self.policy_params, lr=cfg.SAC_LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic_q_params, lr=cfg.SAC_LEARNING_RATE)
        
        self._encoder_frozen = False

        # [FIX 1] Initialize alpha to a lower value
        if cfg.SAC_TARGET_ENTROPY == 'auto':
            self.target_entropy = -float(cfg.N_JOINTS)
        else:
            self.target_entropy = float(cfg.SAC_TARGET_ENTROPY)
        
        # Start with lower alpha to prevent over-exploration
        init_log_alpha = np.log(initial_alpha)
        self.log_alpha = torch.tensor([init_log_alpha], requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.ALPHA_LEARNING_RATE)
        
        # Replay Buffer
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
        
        logger.info(f"Initial alpha: {self.alpha:.4f}")
    
    def _freeze_encoder(self):
        """Freeze shared encoder after warmup."""
        if self._encoder_frozen:
            return
            
        logger.info("="*70)
        logger.info("FREEZING SHARED ENCODER - State Estimator training complete")
        logger.info("="*70)
        
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        
        self._encoder_frozen = True
        
        trainable_actor = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        trainable_critic = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.shared_encoder.parameters())
        logger.info(f"  Trainable actor params: {trainable_actor:,}")
        logger.info(f"  Trainable critic params: {trainable_critic:,}")
        logger.info(f"  Frozen encoder params: {frozen:,}")
        logger.info(f"  Current alpha: {self.alpha:.4f}")
        
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

            self.tb_writer.add_scalar('train/alpha', metrics.get('alpha', self.alpha), step)
            self.tb_writer.add_scalar('train/encoder_frozen', float(self._encoder_frozen), self.total_timesteps)
            self.tb_writer.add_scalars('losses', {
                'actor_loss': metrics.get('actor_loss', 0.0),
                'critic_loss': metrics.get('critic_loss', 0.0),
                'alpha_loss': metrics.get('alpha_loss', 0.0),
                'aux_loss': metrics.get('aux_loss', 0.0)
            }, step)
    
    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().detach().item()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_t, _, _, pred_state_t = self.actor.sample(obs_t, deterministic)
            return action_t.cpu().numpy(), pred_state_t.cpu().numpy()
    
    def validate(self) -> float:
        validation_rewards = []
        logger.info("Starting Validation...")

        for episode in range(cfg.SAC_VAL_EPISODES):
            episode_reward = 0.0
            val_obs, _ = self.val_env.reset()
            
            # Debug stats
            action_magnitudes = []
            remote_q_changes = []
            
            for step in range(cfg.MAX_EPISODE_STEPS):
                obs_batch = val_obs.reshape(1, -1)
                
                # Get Action
                actions_tuple = self.select_action(obs_batch, deterministic=True)
                action = actions_tuple[0][0]
                pred_state = actions_tuple[1][0]
                
                # [DEBUG] Track Action Magnitude
                action_mag = np.linalg.norm(action)
                action_magnitudes.append(action_mag)

                # Step Env
                self.val_env.set_predicted_state(pred_state)
                val_obs, reward, done, truncated, info = self.val_env.step(action)
                
                # [DEBUG] Track Remote Movement
                remote_q = val_obs[:7] # Assuming first 7 are q
                if step > 0:
                    q_change = np.linalg.norm(remote_q - last_q)
                    remote_q_changes.append(q_change)
                last_q = remote_q.copy()
                
                episode_reward += reward
                
                if done or truncated:
                    break
            
            # [DEBUG LOGGING]
            avg_action = np.mean(action_magnitudes)
            avg_movement = np.mean(remote_q_changes) if remote_q_changes else 0.0
            logger.info(f"  Ep {episode}: Avg Action Norm: {avg_action:.4f} | Avg Movement: {avg_movement:.6f}")
            
            validation_rewards.append(episode_reward)
        
        avg_reward = np.mean(validation_rewards)
        logger.info(f"Validation Reward: {avg_reward:.4f} | Alpha: {self.alpha:.4f}")
        return avg_reward
    
    def update_encoder_only(self) -> Dict[str, float]:
        """Stage 1: Train ONLY the encoder, not the critic."""
        if len(self.replay_buffer) < cfg.SAC_BATCH_SIZE:
            return {}

        batch = self.replay_buffer.sample(cfg.SAC_BATCH_SIZE)
        
        # ONLY encoder update (auxiliary loss)
        _, _, _, pred_state = self.actor.sample(batch['remote_states'])
        aux_loss = F.mse_loss(pred_state, batch['true_states'])
        
        self.encoder_optimizer.zero_grad()
        aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder_params, max_norm=1.0)
        self.encoder_optimizer.step()
        
        # NO critic update here!
        
        return {'aux_loss': aux_loss.item()}
    
    def update_policy(self) -> Dict[str, float]:
        """
        [FIX 3] Stage 2: Delayed policy updates (TD3-style).
        Critic updated every step, policy updated every `policy_delay` steps.
        """
        metrics = {}
        if len(self.replay_buffer) < cfg.SAC_BATCH_SIZE:
            return metrics
        
        batch = self.replay_buffer.sample(cfg.SAC_BATCH_SIZE)
        
        # --- 1. Critic Update (Every Step) ---
        with torch.no_grad():
            next_state_t = batch['next_remote_states'] 
            next_actions_t, next_log_probs_t, _, _ = self.actor.sample(next_state_t)
            
            q1_target, q2_target = self.critic_target(next_state_t, next_actions_t)
            q_target_min = torch.min(q1_target, q2_target)
            q_target = batch['rewards'] + (1.0 - batch['dones']) * cfg.SAC_GAMMA * (q_target_min - self.alpha * next_log_probs_t)

        current_state_t = batch['remote_states']
        current_q1, current_q2 = self.critic(current_state_t, batch['actions'])
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_q_params, max_norm=1.0)
        self.critic_optimizer.step()
        
        metrics['critic_loss'] = critic_loss.item()
        
        # --- 2. Delayed Policy Update ---
        if self.num_updates % self.policy_delay == 0:
            for param in self.critic_q_params:
                param.requires_grad = False
                
            actions_t, log_probs_t, _, pred_state = self.actor.sample(current_state_t)
            
            q1_policy, q2_policy = self.critic(current_state_t, actions_t)
            q_policy_min = torch.min(q1_policy, q2_policy)
            
            actor_loss = (self.alpha * log_probs_t - q_policy_min).mean()
            
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            # [FIX 4] Tighter gradient clipping for policy
            torch.nn.utils.clip_grad_norm_(self.policy_params, max_norm=0.5)
            self.policy_optimizer.step()
            
            for param in self.critic_q_params:
                param.requires_grad = True
                
            # Alpha update
            alpha_loss = -(self.log_alpha * (log_probs_t + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            metrics['actor_loss'] = actor_loss.item()
            metrics['alpha_loss'] = alpha_loss.item()
            metrics['alpha'] = self.alpha
            
            # Compute aux loss for logging
            with torch.no_grad():
                aux_loss = F.mse_loss(pred_state, batch['true_states'])
            metrics['aux_loss'] = aux_loss.item()
        
        # --- 3. Soft Update Target Networks ---
        with torch.no_grad():
            for param, target_param in zip(self.critic.q1_net.parameters(), 
                                           self.critic_target.q1_net.parameters()):
                target_param.data.mul_(1.0 - cfg.SAC_TAU)
                target_param.data.add_(cfg.SAC_TAU * param.data)
            for param, target_param in zip(self.critic.q2_net.parameters(), 
                                           self.critic_target.q2_net.parameters()):
                target_param.data.mul_(1.0 - cfg.SAC_TAU)
                target_param.data.add_(cfg.SAC_TAU * param.data)
            
        return metrics
       
    def train(self, total_timesteps: int):
        self._init_tensorboard()
        self.training_start_time = datetime.now()
        
        metrics = {}
        logger.info("="*70)
        logger.info("Starting SAC Training with SHARED ENCODER (Fixed Version)")
        logger.info(f"  Obs Dim: {cfg.OBS_DIM} | Output Dim: {cfg.N_JOINTS}")
        logger.info(f"  Stage 1 (Encoder + Critic Warmup): {cfg.SAC_START_STEPS} steps")
        logger.info(f"  Stage 2 (RL with Frozen Encoder, Delayed Policy): remaining steps")
        logger.info(f"  Policy Delay: {self.policy_delay}")
        logger.info(f"  Initial Alpha: {self.alpha:.4f}")
        logger.info("="*70)
        
        episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        completed_episode_rewards = deque(maxlen=100)
        
        obs = self.env.reset()
        
        for t in range(int(total_timesteps // self.num_envs)):
            
            # 1. Action Selection
            if self.total_timesteps < cfg.SAC_START_STEPS:
                # Random exploration during warmup
                actions_batch = np.array([
                    np.random.uniform(-1.0, 1.0, size=(cfg.N_JOINTS,)) 
                    for _ in range(self.num_envs)
                ])
                actions_batch = actions_batch * cfg.MAX_TORQUE_COMPENSATION
                
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    _, _, _, pred_state_t = self.actor.sample(obs_t)
                    pred_states_batch = pred_state_t.cpu().numpy()
            else:
                actions_batch, pred_states_batch = self.select_action(obs)

            # 2. Step Environment
            next_obs, rewards_batch, dones_batch, infos_batch = self.env.step(actions_batch)
            
            # 3. Store Experience
            for i in range(self.num_envs):
                if infos_batch[i].get('is_in_warmup', False):
                    continue
                true_state = infos_batch[i].get('true_state', np.zeros(cfg.N_JOINTS * 2, dtype=np.float32))
                self.replay_buffer.add(
                    obs[i],
                    actions_batch[i],
                    rewards_batch[i],
                    next_obs[i],
                    dones_batch[i],
                    true_state
                )

                pred_error = np.linalg.norm(true_state - pred_states_batch[i])
                self.prediction_error_history.append(pred_error)
                self.real_time_error_history.append(infos_batch[i].get('real_time_joint_error', np.nan))
                self.delay_steps_history.append(infos_batch[i].get('current_delay_steps', np.nan))

            obs = next_obs
            self.total_timesteps += self.num_envs
            episode_rewards += rewards_batch

            for i in range(self.num_envs):
                if dones_batch[i]:
                    completed_episode_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0

            # 4. Training Updates
            if len(self.replay_buffer) >= cfg.SAC_BATCH_SIZE:
                
                if self.total_timesteps < cfg.SAC_START_STEPS:
                    # [FIX] Stage 1: Train encoder AND critic
                    for _ in range(int(cfg.SAC_UPDATES_PER_STEP * self.num_envs)):
                        metrics = self.update_encoder_only()
                        
                    if (t + 1) % 1000 == 0:
                        logger.info(f"[Stage 1 - Warmup {self.total_timesteps}/{cfg.SAC_START_STEPS}] "
                                   f"Aux Loss: {metrics.get('aux_loss', 0.0):.6f} | "
                                   f"Critic Loss: {metrics.get('critic_loss', 0.0):.4f}")
                else:
                    if not self._encoder_frozen:
                        self._freeze_encoder()
                    
                    for _ in range(int(cfg.SAC_UPDATES_PER_STEP * self.num_envs)):
                        metrics = self.update_policy()
                        self.num_updates += 1

            # Validation
            if (t + 1) % (cfg.SAC_VAL_FREQ // self.num_envs) == 0 and self.total_timesteps >= cfg.SAC_START_STEPS:
                val_reward = self.validate()
                self.validation_rewards_history.append(val_reward)
                if val_reward > self.best_validation_reward:
                    self.best_validation_reward = val_reward
                    self.patience_counter = 0
                    self.save_checkpoint("best_policy.pth")
                else:
                    self.patience_counter += 1

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
            'shared_encoder_state_dict': self.shared_encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'log_alpha': self.log_alpha,
            'total_timesteps': self.total_timesteps,
            'best_validation_reward': self.best_validation_reward,
            'encoder_frozen': self._encoder_frozen,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")