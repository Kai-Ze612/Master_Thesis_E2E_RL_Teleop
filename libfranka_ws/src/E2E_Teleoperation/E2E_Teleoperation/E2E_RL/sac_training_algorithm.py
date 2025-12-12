"""
SAC Training Algorithm - Version 20: Standard Scale (Final)

CHANGES:
1. [FIX] Retuned weights for Reward Scale = 1.0.
   - alpha: 0.01 - 0.05 (Small entropy bonus)
   - sac_weight: 1.0
   - bc_weight: 10.0 (Dominate anchor)
2. This creates a standard, stable RL setup.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import copy
import os
import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from E2E_Teleoperation.E2E_RL.sac_policy_network import SharedLSTMEncoder, JointActor, JointCritic
import E2E_Teleoperation.config.robot_config as cfg

logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    # Stable parameters
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    encoder_lr: float = 5e-4
    gamma: float = 0.99
    tau: float = 0.001
    
    # [FIX] Lower Alpha for 0-1 Reward Scale
    initial_alpha: float = 0.01
    alpha_min: float = 0.001
    alpha_max: float = 0.05
    target_entropy_scale: float = 0.5
    fixed_alpha: bool = False
    
    encoder_warmup_steps: int = 10000
    teacher_warmup_steps: int = 30000
    buffer_size: int = 1000000
    batch_size: int = 256
    policy_delay: int = 2
    validation_freq: int = 5000
    checkpoint_freq: int = 50000
    log_freq: int = 1000


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, true_state_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.is_demo = np.zeros((capacity, 1), dtype=np.float32)
        self.true_states = np.zeros((capacity, true_state_dim), dtype=np.float32)
    def add(self, obs, action, reward, next_obs, done, true_state, is_demo=False):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.is_demo[self.ptr] = float(is_demo)
        self.true_states[self.ptr] = true_state
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.FloatTensor(self.obs[idxs]).to(self.device),
            'actions': torch.FloatTensor(self.actions[idxs]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[idxs]).to(self.device),
            'next_obs': torch.FloatTensor(self.next_obs[idxs]).to(self.device),
            'dones': torch.FloatTensor(self.dones[idxs]).to(self.device),
            'is_demo': torch.FloatTensor(self.is_demo[idxs]).to(self.device),
            'true_states': torch.FloatTensor(self.true_states[idxs]).to(self.device),
        }
    def __len__(self): return self.size


class SACTrainer:
    def __init__(self, env, policy_network: JointActor, config: SACConfig = None, output_dir: str = "./rl_training_output", device: torch.device = None):
        self.env = env
        self.config = config or SACConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.num_envs = env.num_envs
        self.obs_dim = cfg.OBS_DIM
        self.action_dim = cfg.N_JOINTS
        self.true_state_dim = 14
        self.actor = policy_network.to(self.device)
        self.shared_encoder = self.actor.encoder
        self.critic = JointCritic(shared_encoder=self.shared_encoder, action_dim=self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters(): param.requires_grad = False
        self.log_alpha = torch.tensor(np.log(self.config.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.target_entropy = -self.action_dim * self.config.target_entropy_scale
        self._build_optimizers()
        self.replay_buffer = ReplayBuffer(self.config.buffer_size, self.obs_dim, self.action_dim, self.true_state_dim, self.device)
        self.total_timesteps = 0
        self.update_count = 0
        self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard_sac"))
        if hasattr(cfg, 'MAX_TORQUE_COMPENSATION'):
            self.action_scale = np.array(cfg.MAX_TORQUE_COMPENSATION, dtype=np.float32)
        else:
            self.action_scale = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 1.0], dtype=np.float32)
        logger.info(f"Initial alpha: {self.alpha:.4f}")

    def _build_optimizers(self):
        self.encoder_optimizer = optim.Adam(self.shared_encoder.parameters(), lr=self.config.encoder_lr)
        self.actor_optimizer = optim.Adam([p for n, p in self.actor.named_parameters() if 'encoder' not in n], lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam([p for n, p in self.critic.named_parameters() if 'encoder' not in n], lr=self.config.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
    @property
    def alpha(self): return self.log_alpha.exp().item()
    def _freeze_encoder(self):
        for param in self.shared_encoder.parameters(): param.requires_grad = False
        logger.info("FREEZING SHARED ENCODER")
    def _set_teacher_mode(self, val): self.env.env_method("set_teacher_mode", val)
    def _set_ground_truth_mode(self, val): self.env.env_method("set_ground_truth_mode", val)
    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            if obs_tensor.dim() == 1: obs_tensor = obs_tensor.unsqueeze(0)
            action, _, _, pred = self.actor.sample(obs_tensor, deterministic=deterministic)
            return action.cpu().numpy(), pred.cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor], update_policy: bool = True):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        true_states = batch['true_states']
        is_demo = batch['is_demo']
        metrics = {}
        if self.total_timesteps < self.config.encoder_warmup_steps:
            target_hist = obs[:, -(cfg.RNN_SEQUENCE_LENGTH * cfg.ESTIMATOR_STATE_DIM):]
            history = target_hist.view(-1, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
            _, pred = self.shared_encoder(history)
            true_q = true_states[:, :cfg.N_JOINTS]
            true_qd = true_states[:, cfg.N_JOINTS:]
            q_norm = (true_q - torch.tensor(cfg.Q_MEAN, device=self.device)) / torch.tensor(cfg.Q_STD, device=self.device)
            qd_norm = (true_qd - torch.tensor(cfg.QD_MEAN, device=self.device)) / torch.tensor(cfg.QD_STD, device=self.device)
            target = torch.cat([q_norm, qd_norm], dim=1)
            aux_loss = F.mse_loss(pred, target)
            self.encoder_optimizer.zero_grad()
            aux_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.shared_encoder.parameters(), 1.0)
            self.encoder_optimizer.step()
            metrics['aux_loss'] = aux_loss.item()
            return metrics

        with torch.no_grad():
            next_act, next_lp, _, _ = self.actor.sample(next_obs)
            nq1, nq2 = self.critic_target(next_obs, next_act)
            min_nq = torch.min(nq1, nq2) - self.alpha * next_lp
            target_q = rewards + (1 - dones) * self.config.gamma * min_nq

        cq1, cq2 = self.critic(obs, actions)
        critic_loss = F.smooth_l1_loss(cq1, target_q) + F.smooth_l1_loss(cq2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for n, p in self.critic.named_parameters() if 'encoder' not in n], 1.0)
        self.critic_optimizer.step()
        metrics['critic_loss'] = critic_loss.item()
        metrics['q1_mean'] = cq1.mean().item()

        if update_policy:
            new_act, log_prob, raw_act, _ = self.actor.sample(obs)
            q1_new, q2_new = self.critic(obs, new_act)
            q_min = torch.min(q1_new, q2_new)
            sac_loss = (self.alpha * log_prob - q_min).mean()
            
            is_teacher_phase = self.total_timesteps < self.config.teacher_warmup_steps
            sac_weight = 0.0 if is_teacher_phase else 1.0 
            bc_weight = 100.0 if is_teacher_phase else 10.0 # [FIX] Adjusted for Unit Reward
            
            bc_loss = 0.0
            if is_demo.sum() > 0:
                demo_mask = (is_demo > 0.5).flatten()
                if demo_mask.any():
                    bc_error = F.mse_loss(new_act[demo_mask], actions[demo_mask])
                    bc_loss = bc_error * bc_weight
            
            logit_reg_loss = (raw_act**2).mean() * 0.001
            actor_loss = (sac_weight * sac_loss) + bc_loss + logit_reg_loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for n, p in self.actor.named_parameters() if 'encoder' not in n], 1.0)
            self.actor_optimizer.step()
            
            metrics['actor_loss'] = actor_loss.item()
            metrics['bc_loss'] = bc_loss.item() if isinstance(bc_loss, torch.Tensor) else 0.0
            
            if not self.config.fixed_alpha and not is_teacher_phase:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    self.log_alpha.data = torch.clamp(self.log_alpha.data, min=np.log(self.config.alpha_min), max=np.log(self.config.alpha_max))
                metrics['alpha_loss'] = alpha_loss.item()
            metrics['alpha'] = self.alpha
            self._soft_update_targets()
        return metrics

    def _soft_update_targets(self):
        tau = self.config.tau
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def train(self, total_timesteps: int):
        logger.info("Starting SAC Training v20")
        obs = self.env.reset()[0]
        encoder_frozen = False
        pure_rl_started = False
        self._set_teacher_mode(True)
        self._set_ground_truth_mode(True)
        while self.total_timesteps < total_timesteps:
            if not encoder_frozen and self.total_timesteps >= self.config.encoder_warmup_steps:
                self._freeze_encoder()
                self._set_ground_truth_mode(False)
                encoder_frozen = True
            if not pure_rl_started and self.total_timesteps >= self.config.teacher_warmup_steps:
                self._set_teacher_mode(False)
                pure_rl_started = True
            actions, pred = self.select_action(obs)
            for i in range(self.num_envs): self.env.env_method("set_predicted_state", pred[i], indices=i)
            next_obs, rewards, dones, infos = self.env.step(actions)
            for i in range(self.num_envs):
                is_demo = (self.total_timesteps < self.config.teacher_warmup_steps)
                actual_action = infos[i].get('actual_action', actions[i])
                clipped_action = np.clip(actual_action, -self.action_scale, self.action_scale)
                self.replay_buffer.add(obs[i], clipped_action, rewards[i], next_obs[i], float(dones[i]), infos[i]['true_state'], is_demo)
            obs = next_obs
            self.total_timesteps += self.num_envs
            if len(self.replay_buffer) >= self.config.batch_size:
                batch = self.replay_buffer.sample(self.config.batch_size)
                update_policy = (self.update_count % self.config.policy_delay == 0) and encoder_frozen
                metrics = self.update(batch, update_policy)
                self.update_count += 1
                if self.total_timesteps % self.config.log_freq == 0:
                    phase = "Stage1" if not encoder_frozen else "Stage2_BC" if not pure_rl_started else "Stage3_SAC"
                    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    logger.info(f"[{phase}] Step {self.total_timesteps} | {metrics_str}")
            if self.total_timesteps % self.config.validation_freq == 0 and encoder_frozen:
                val = self.validate()
                logger.info(f"Validation Step {self.total_timesteps} | Reward: {val['reward']:.2f}")
                self.writer.add_scalar("val/reward", val['reward'], self.total_timesteps)
            if self.total_timesteps % self.config.checkpoint_freq == 0:
                self.save_checkpoint(f"checkpoint_{self.total_timesteps}.pth")
        self.save_checkpoint("final_policy.pth")

    def validate(self, num_episodes: int = 5) -> Dict[str, float]:
        was_teacher = (self.total_timesteps < self.config.teacher_warmup_steps)
        self._set_teacher_mode(False)
        self._set_ground_truth_mode(False)
        total_rewards = []
        for _ in range(num_episodes):
            obs = self.env.reset()[0]
            ep_reward = 0
            done = False
            while not done:
                action, pred_state = self.select_action(obs, deterministic=True)
                for i in range(self.num_envs): self.env.env_method("set_predicted_state", pred_state[i], indices=i)
                next_obs, reward, done_arr, _ = self.env.step(action)
                ep_reward += reward[0]
                done = done_arr[0]
                obs = next_obs
            total_rewards.append(ep_reward)
        self._set_teacher_mode(was_teacher)
        self._set_ground_truth_mode(self.total_timesteps < self.config.encoder_warmup_steps)
        return {'reward': np.mean(total_rewards)}

    def save_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def load_checkpoint(self, path):
        pass