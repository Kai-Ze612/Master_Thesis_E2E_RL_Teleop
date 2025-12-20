import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, Optional
import copy

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.sac_policy_network import (
    ContinuousLSTMEncoder, JointActor, JointCritic, create_actor_critic
)

logger = logging.getLogger(__name__)


class EarlyStopper:
    """Early stopping based on validation reward."""
    def __init__(self, patience: int = cfg.EARLY_STOP_PATIENCE, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = -np.inf
        self.best_step = 0
    
    def check(self, metric: float, current_step: int = 0) -> Tuple[bool, bool]:
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.best_step = current_step
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping triggered. Best metric: {self.best_metric:.2f} "
                    f"at step {self.best_step}. No improvement for {self.patience} evaluations."
                )
                return True, False
            return False, False
    
    def reset(self):
        self.counter = 0
        self.best_metric = -np.inf
        self.best_step = 0


class Trainer:
    def __init__(self, env, output_dir: str):
        self.env = env
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))
        
        # Networks
        self.shared_encoder, self.actor, self.critic = create_actor_critic(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.enc_optimizer = optim.Adam(self.shared_encoder.parameters(), lr=cfg.ENCODER_LR)
        self.actor_optimizer = optim.Adam(
            [p for n, p in self.actor.named_parameters() if 'encoder' not in n],
            lr=cfg.ACTOR_LR
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR)
        
        # SAC entropy
        self.target_entropy = getattr(cfg, 'TARGET_ENTROPY', -cfg.N_JOINTS)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.ALPHA_LR)
        
        # Buffer
        self.buffer = RecoveryBuffer(cfg.BUFFER_SIZE, cfg.OBS_DIM, cfg.N_JOINTS, self.device)
        
        # Early stopping
        self.stopper = EarlyStopper(patience=cfg.EARLY_STOP_PATIENCE, min_delta=1.0)
        
        # Training state
        self.global_step = 0
        
    def _get_alpha(self):
        return self.log_alpha.exp()

    # =========================================================================
    # STAGE 1: Encoder Pre-training
    # =========================================================================
    def train_stage_1_encoder(self):
        logger.info("=" * 60)
        logger.info(">>> STAGE 1: ENCODER PRE-TRAINING")
        logger.info("=" * 60)
        
        # Freeze policy layers
        for param in self.actor.backbone.parameters(): param.requires_grad = False
        for param in self.actor.fc_mean.parameters(): param.requires_grad = False
        for param in self.actor.fc_log_std.parameters(): param.requires_grad = False
        
        updates_completed = 0
        
        while updates_completed < cfg.STAGE1_STEPS:
            logger.info(f"Collecting {cfg.STAGE1_COLLECTION_STEPS} steps for encoder training...")
            self.collect_with_noise_injection(steps=cfg.STAGE1_COLLECTION_STEPS, noise_scale=0.0, noise_type='gaussian')
            
            logger.info(f"Training encoder...")
            for _ in range(cfg.STAGE1_COLLECTION_STEPS):
                if updates_completed >= cfg.STAGE1_STEPS: break
                
                batch = self.buffer.sample_with_recovery_weight(cfg.BATCH_SIZE, recovery_weight=1.0)
                obs = batch['obs']
                
                # Forward pass
                target_history = obs[:, -cfg.TARGET_HISTORY_DIM:]
                history_seq = target_history.view(-1, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
                _, pred_state, _ = self.shared_encoder(history_seq)
                
                # Ground truth
                idxs = np.random.randint(0, self.buffer.size, size=cfg.BATCH_SIZE)
                true_states = torch.FloatTensor(self.buffer.true_states[idxs]).to(self.device)
                
                # Normalize targets
                t_q = true_states[:, :7]
                t_qd = true_states[:, 7:]
                t_q_norm = (t_q - torch.tensor(cfg.Q_MEAN, device=self.device)) / torch.tensor(cfg.Q_STD, device=self.device)
                t_qd_norm = (t_qd - torch.tensor(cfg.QD_MEAN, device=self.device)) / torch.tensor(cfg.QD_STD, device=self.device)
                target = torch.cat([t_q_norm, t_qd_norm], dim=1)
                
                loss = F.mse_loss(pred_state, target)
                
                self.enc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.shared_encoder.parameters(), max_norm=cfg.MAX_GRAD_NORM)
                self.enc_optimizer.step()
                
                updates_completed += 1
                if updates_completed % cfg.LOG_INTERVAL == 0:
                    logger.info(f"[Stage 1] Update {updates_completed}/{cfg.STAGE1_STEPS} | Loss: {loss.item():.6f}")
                    self.writer.add_scalar("Stage1/EncoderLoss", loss.item(), updates_completed)
        
        for param in self.actor.parameters(): param.requires_grad = True
        logger.info(">>> STAGE 1 COMPLETE")
        self.save_checkpoint("stage1_complete.pth")

    # =========================================================================
    # STAGE 2: BC & DAgger
    # =========================================================================
    def collect_with_noise_injection(self, steps: int, noise_scale: float = 2.0, noise_type: str = 'gaussian'):
        obs, info = self.env.reset()
        hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
        prev_prediction = torch.zeros(1, 14, device=self.device)
        episodes_completed = 0
        
        for i in range(steps):
            has_new_obs = info.get('has_new_obs', True)
            r_q, r_qd = self.env.remote.get_joint_state()
            t_q, t_qd = self.env.leader_hist[-1]
            teacher_action = self.env._compute_teacher_torque(r_q, r_qd, t_q, t_qd)
            
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_scale, size=teacher_action.shape)
            else:
                noise = np.zeros_like(teacher_action)
            
            driving_action = np.clip(teacher_action + noise, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
            
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, _, pred, next_hidden = self.actor.sample(
                    obs_t, hidden=hidden, prev_prediction=prev_prediction,
                    has_new_obs=has_new_obs, deterministic=True
                )
            
            next_obs, reward, terminated, truncated, next_info = self.env.step(driving_action)
            done = terminated or truncated
            
            self.buffer.add(
                obs=obs, action=driving_action, reward=reward, next_obs=next_obs, done=done,
                teacher_action=teacher_action, true_state=next_info['true_state'],
                has_new_obs=has_new_obs, prev_prediction=prev_prediction.cpu().numpy()[0],
                is_recovery=terminated
            )
            
            obs = next_obs; info = next_info; hidden = next_hidden; prev_prediction = pred
            
            if done:
                episodes_completed += 1
                obs, info = self.env.reset()
                hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
                prev_prediction = torch.zeros(1, 14, device=self.device)
        
        logger.info(f"Collected {steps} steps, noise={noise_scale:.1f}Nm, {episodes_completed} episodes")

    def train_bc_with_recovery_weighting(self, num_updates: int, recovery_weight: float):
        total_loss = 0.0
        for _ in range(num_updates):
            batch = self.buffer.sample_with_recovery_weight(cfg.BATCH_SIZE, recovery_weight=recovery_weight)
            obs = batch['obs']
            teacher_actions = batch['teacher_actions']
            sample_weights = batch['weights']
            
            student_action, _, _, _, _ = self.actor.sample(obs, hidden=None, prev_prediction=None, has_new_obs=True, deterministic=True)
            
            scale = self.actor.action_scale
            per_sample_loss = F.mse_loss(student_action / scale, teacher_actions / scale, reduction='none').mean(dim=1)
            weighted_loss = (per_sample_loss * sample_weights).mean()
            
            self.actor_optimizer.zero_grad()
            self.enc_optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=cfg.MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(self.shared_encoder.parameters(), max_norm=cfg.MAX_GRAD_NORM)
            self.actor_optimizer.step()
            self.enc_optimizer.step()
            total_loss += weighted_loss.item()
        return total_loss / num_updates

    def train_stage_2_with_recovery(self):
        logger.info("=" * 60); logger.info(">>> STAGE 2: BC WITH RECOVERY LEARNING"); logger.info("=" * 60)
        self.global_step = 0
        
        # Phase 2a: Noise Injection
        while self.global_step < cfg.STAGE2A_TOTAL_STEPS:
            noise_scale = 0.5
            for threshold, scale in cfg.STAGE2_NOISE_SCHEDULE:
                if self.global_step >= threshold: noise_scale = scale
            
            logger.info(f"Step {self.global_step} | Noise injection, scale={noise_scale:.1f}Nm")
            self.collect_with_noise_injection(steps=cfg.STAGE2_COLLECTION_STEPS, noise_scale=noise_scale)
            self.global_step += cfg.STAGE2_COLLECTION_STEPS
            
            avg_loss = self.train_bc_with_recovery_weighting(num_updates=cfg.STAGE2_COLLECTION_STEPS, recovery_weight=cfg.STAGE2_RECOVERY_WEIGHT)
            logger.info(f"[Phase 2a] Avg BC Loss: {avg_loss:.6f}")
            self.writer.add_scalar("Stage2a/BCLoss", avg_loss, self.global_step)
            
            if self.global_step % cfg.VAL_FREQ == 0:
                val_reward = self.validate()
                logger.info(f"[Validation] Reward: {val_reward:.2f}")
                should_stop, is_best = self.stopper.check(val_reward, self.global_step)
                if is_best: self.save_checkpoint("best_policy_phase2a.pth")
                if should_stop: break
        
        self.stopper.reset()
        
        # Phase 2b: DAgger (Clean collection)
        logger.info("--- Phase 2b: DAgger with Recovery Focus ---")
        while self.global_step < cfg.STAGE2_TOTAL_STEPS:
            self.collect_with_noise_injection(steps=cfg.STAGE2_COLLECTION_STEPS, noise_scale=0.0)
            self.global_step += cfg.STAGE2_COLLECTION_STEPS
            avg_loss = self.train_bc_with_recovery_weighting(num_updates=cfg.STAGE2_COLLECTION_STEPS, recovery_weight=cfg.STAGE2_RECOVERY_WEIGHT)
            if self.global_step % cfg.VAL_FREQ == 0:
                val_reward = self.validate()
                logger.info(f"[Validation] Reward: {val_reward:.2f}")
                should_stop, is_best = self.stopper.check(val_reward, self.global_step)
                if is_best: self.save_checkpoint("best_policy.pth")
                if should_stop: break
        logger.info(">>> STAGE 2 COMPLETE")

    # =========================================================================
    # STAGE 3: SAC Fine-tuning
    # =========================================================================
    def train_stage_3_sac(self):
        logger.info("=" * 60)
        logger.info(">>> STAGE 3: SAC FINE-TUNING (FROZEN ENCODER + RESIDUAL RL)")
        logger.info("=" * 60)
        
        self.stopper.reset()
        
        # --- FIX 1: FREEZE THE ENCODER ---
        logger.info("Freezing Shared Encoder for Stage 3...")
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
            
        # --- FIX 2: VARIANCE SUPPRESSION ---
        with torch.no_grad():
            self.actor.fc_log_std.bias.fill_(-5.0)
            self.actor.fc_log_std.weight.fill_(0.01)
            self.log_alpha.fill_(np.log(0.001))

        # Reduce learning rates
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = cfg.ACTOR_LR * cfg.S3_LR_SCALE
        for param_group in self.alpha_optimizer.param_groups:
            param_group['lr'] = cfg.ALPHA_LR * cfg.S3_ALPHA_LR_SCALE
            
        # --- FIX 3: CRITIC WARM-UP ---
        logger.info(f"Phase 3a: Critic Warm-up ({cfg.S3_CRITIC_WARMUP_STEPS} steps)...")
        obs, info = self.env.reset()
        hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
        prev_prediction = torch.zeros(1, 14, device=self.device)
        ep_reward = 0
        
        for warmup_step in range(cfg.S3_CRITIC_WARMUP_STEPS):
            has_new_obs = info.get('has_new_obs', True)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Teacher Drives
            r_q, r_qd = self.env.remote.get_joint_state()
            t_q, t_qd = self.env.leader_hist[-1]
            teacher_action = self.env._compute_teacher_torque(r_q, r_qd, t_q, t_qd)
            
            with torch.no_grad():
                _, _, _, pred, next_hidden = self.actor.sample(
                    obs_t, hidden=hidden, prev_prediction=prev_prediction,
                    has_new_obs=has_new_obs, deterministic=False
                )
            
            driving_action = teacher_action
            next_obs, reward, terminated, truncated, next_info = self.env.step(driving_action)
            done = terminated or truncated
            ep_reward += reward

            self.buffer.add(
                obs=obs, action=driving_action, 
                reward=reward * cfg.S3_REWARD_SCALE, 
                next_obs=next_obs, done=done, 
                teacher_action=teacher_action, true_state=next_info['true_state'],
                has_new_obs=has_new_obs, prev_prediction=prev_prediction.cpu().numpy()[0],
                is_recovery=terminated
            )
            
            obs = next_obs; info = next_info; hidden = next_hidden; prev_prediction = pred
            
            if done:
                obs, info = self.env.reset()
                hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
                prev_prediction = torch.zeros(1, 14, device=self.device)
                ep_reward = 0

            if self.buffer.current_size > cfg.BATCH_SIZE:
                self._update_critic_only_frozen_encoder()
                
            if warmup_step % 5000 == 0:
                logger.info(f"Warmup Step {warmup_step}/{cfg.S3_CRITIC_WARMUP_STEPS}")

        logger.info(">>> Critic Warm-up Complete.")

        # --- 4. FULL SAC TRAINING ---
        ep_reward = 0
        stage3_step_counter = 0 
        obs, info = self.env.reset()
        hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
        prev_prediction = torch.zeros(1, 14, device=self.device)
        
        debug_sac_loss = 0.0
        debug_bc_loss = 0.0
        
        while self.global_step < cfg.STAGE3_TOTAL_STEPS:
            self.global_step += 1
            stage3_step_counter += 1
            
            # BC Decay
            progress = max(0, min(1, stage3_step_counter / cfg.S3_BC_DECAY_STEPS))
            bc_weight = cfg.S3_INITIAL_BC_WEIGHT - (progress * (cfg.S3_INITIAL_BC_WEIGHT - cfg.S3_MIN_BC_WEIGHT))
            
            has_new_obs = info.get('has_new_obs', True)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _, _, pred, next_hidden = self.actor.sample(
                    obs_t, hidden=hidden, prev_prediction=prev_prediction,
                    has_new_obs=has_new_obs, deterministic=False
                )
            
            driving_action = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, next_info = self.env.step(driving_action)
            done = terminated or truncated
            
            r_q, r_qd = self.env.remote.get_joint_state()
            t_q, t_qd = self.env.leader_hist[-1]
            teacher_action = self.env._compute_teacher_torque(r_q, r_qd, t_q, t_qd)

            self.buffer.add(
                obs=obs, action=driving_action, 
                reward=reward * cfg.S3_REWARD_SCALE, 
                next_obs=next_obs, done=done, 
                teacher_action=teacher_action, true_state=next_info['true_state'],
                has_new_obs=has_new_obs, prev_prediction=prev_prediction.cpu().numpy()[0],
                is_recovery=terminated
            )
            
            obs = next_obs; info = next_info; hidden = next_hidden; prev_prediction = pred
            ep_reward += reward
            
            if self.buffer.current_size > cfg.BATCH_SIZE:
                loss_info = self._update_parameters_sac_frozen_encoder(bc_weight)
                debug_sac_loss = loss_info['a_loss']
                debug_bc_loss = loss_info['bc_loss']
                
                if self.global_step % cfg.LOG_INTERVAL == 0:
                    self.writer.add_scalar("Stage3/CriticLoss", loss_info['c_loss'], self.global_step)
                    self.writer.add_scalar("Stage3/ActorLoss", loss_info['a_loss'], self.global_step)
                    self.writer.add_scalar("Stage3/BCLoss", loss_info['bc_loss'], self.global_step)
                    self.writer.add_scalar("Stage3/BCWeight", bc_weight, self.global_step)

            if done:
                weighted_bc = bc_weight * debug_bc_loss
                est_q = -debug_sac_loss 
                logger.info(f"Step {self.global_step} | R: {ep_reward:.0f} | Est.Q: {est_q:.2f} | BC Loss: {weighted_bc:.2f}")
                self.writer.add_scalar("Stage3/EpReward", ep_reward, self.global_step)
                
                obs, info = self.env.reset()
                hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
                prev_prediction = torch.zeros(1, 14, device=self.device)
                ep_reward = 0
                
            if self.global_step % cfg.VAL_FREQ == 0:
                val_reward = self.validate()
                logger.info(f"[Stage 3 Validation] Reward: {val_reward:.2f}")
                self.writer.add_scalar("Validation/Reward", val_reward, self.global_step)
                should_stop, is_best = self.stopper.check(val_reward, self.global_step)
                if is_best: self.save_checkpoint("best_policy_stage3.pth")

        logger.info(">>> STAGE 3 COMPLETE")

    # =========================================================================
    # STAGE 3 HELPERS (FROZEN ENCODER)
    # =========================================================================
    def _update_critic_only_frozen_encoder(self):
        batch = self.buffer.sample_with_recovery_weight(cfg.BATCH_SIZE, recovery_weight=1.0)
        obs, next_obs = batch['obs'], batch['next_obs']
        actions, rewards, dones = batch['actions'], batch['rewards'], batch['dones']
        
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = self.actor.sample(
                next_obs, hidden=None, prev_prediction=None, has_new_obs=True, deterministic=False
            )
            alpha = 0.001 
            target_q1, target_q2, _ = self.critic_target(next_obs, next_action, hidden=None, has_new_obs=True)
            min_target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_q = rewards + (1 - dones) * cfg.GAMMA * min_target_q

        current_q1, current_q2, _ = self.critic(obs, actions, hidden=None, has_new_obs=True)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
        self.critic_optimizer.step()

    def _update_parameters_sac_frozen_encoder(self, bc_weight: float):
        alpha = self._get_alpha()
        batch = self.buffer.sample_with_recovery_weight(cfg.BATCH_SIZE, recovery_weight=1.0)
        
        obs, next_obs = batch['obs'], batch['next_obs']
        actions, rewards, dones = batch['actions'], batch['rewards'], batch['dones']
        teacher_actions = batch['teacher_actions']

        # 1. Critic Update
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = self.actor.sample(
                next_obs, hidden=None, has_new_obs=True, deterministic=False
            )
            target_q1, target_q2, _ = self.critic_target(next_obs, next_action, hidden=None, has_new_obs=True)
            min_target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_q = rewards + (1 - dones) * cfg.GAMMA * min_target_q

        current_q1, current_q2, _ = self.critic(obs, actions, hidden=None, has_new_obs=True)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        # 2. Actor Update
        new_action, log_prob, _, _, _ = self.actor.sample(obs, hidden=None, has_new_obs=True, deterministic=False)
        
        q1_pi, q2_pi, _ = self.critic(obs, new_action, hidden=None, has_new_obs=True)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        sac_loss = (alpha * log_prob - min_q_pi).mean()
        bc_loss = F.mse_loss(new_action, teacher_actions)
        total_actor_loss = sac_loss + (bc_weight * bc_loss)

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        # 3. Alpha Update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 4. Polyak
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1 - cfg.POLYAK_TAU)
                target_param.data.add_(cfg.POLYAK_TAU * param.data)
                
        return {'c_loss': critic_loss.item(), 'a_loss': sac_loss.item(), 'bc_loss': bc_loss.item(), 'alpha': alpha.item()}

    # =========================================================================
    # Helpers
    # =========================================================================
    def validate(self) -> float:
        total_reward = 0.0
        self.actor.eval()
        for ep_idx in range(cfg.VAL_EPISODES):
            obs, info = self.env.reset()
            hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
            prev_prediction = torch.zeros(1, 14, device=self.device)
            ep_reward = 0.0
            while True:
                has_new_obs = info.get('has_new_obs', True)
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _, pred, next_hidden = self.actor.sample(
                        obs_t, hidden=hidden, prev_prediction=prev_prediction,
                        has_new_obs=has_new_obs, deterministic=True
                    )
                action_np = action.cpu().numpy()[0]
                next_obs, reward, terminated, truncated, next_info = self.env.step(action_np)
                done = terminated or truncated
                ep_reward += reward
                obs = next_obs; info = next_info; hidden = next_hidden; prev_prediction = pred
                if done: break
            total_reward += ep_reward
        self.actor.train()
        return total_reward / cfg.VAL_EPISODES

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'encoder': self.shared_encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'global_step': self.global_step,
        }, path)
        logger.info(f"Saved: {path}")

    def load_checkpoint(self, checkpoint_path: str, load_critic: bool = False):
        if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.shared_encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        if load_critic and 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic'])
            logger.info("Critic loaded.")
        else:
            logger.info("Critic NOT loaded (fresh start for SAC).")
        logger.info("Checkpoint loaded successfully.")


class RecoveryBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity; self.device = device; self.ptr = 0; self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.teacher_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.true_states = np.zeros((capacity, 14), dtype=np.float32)
        self.has_new_obs = np.zeros((capacity, 1), dtype=np.float32)
        self.prev_predictions = np.zeros((capacity, 14), dtype=np.float32)
        self.is_recovery = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, teacher_action, true_state, has_new_obs, prev_prediction, is_recovery: bool = False):
        self.obs[self.ptr] = obs; self.actions[self.ptr] = action; self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs; self.dones[self.ptr] = float(done)
        self.teacher_actions[self.ptr] = teacher_action; self.true_states[self.ptr] = true_state
        self.has_new_obs[self.ptr] = float(has_new_obs); self.prev_predictions[self.ptr] = prev_prediction
        self.is_recovery[self.ptr] = float(is_recovery)
        self.ptr = (self.ptr + 1) % self.capacity; self.size = min(self.size + 1, self.capacity)

    def sample_with_recovery_weight(self, batch_size: int, recovery_weight: float = 2.0) -> Dict[str, torch.Tensor]:
        weights = np.ones(self.size)
        recovery_mask = self.is_recovery[:self.size, 0] > 0.5
        weights[recovery_mask] = recovery_weight
        weights = weights / weights.sum()
        idxs = np.random.choice(self.size, size=batch_size, p=weights)
        sample_weights = np.ones(batch_size)
        sample_weights[self.is_recovery[idxs, 0] > 0.5] = recovery_weight
        return {
            'obs': torch.FloatTensor(self.obs[idxs]).to(self.device),
            'next_obs': torch.FloatTensor(self.next_obs[idxs]).to(self.device),
            'actions': torch.FloatTensor(self.actions[idxs]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[idxs]).to(self.device),
            'dones': torch.FloatTensor(self.dones[idxs]).to(self.device),
            'teacher_actions': torch.FloatTensor(self.teacher_actions[idxs]).to(self.device),
            'weights': torch.FloatTensor(sample_weights).to(self.device),
            'is_recovery': torch.FloatTensor(self.is_recovery[idxs]).to(self.device),
        }

    def get_recovery_ratio(self) -> float:
        if self.size == 0: return 0.0
        return self.is_recovery[:self.size].mean()
    
    @property
    def current_size(self): return self.size