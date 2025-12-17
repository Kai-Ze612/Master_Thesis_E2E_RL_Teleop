"""
SAC Training Algorithm with Continuous LSTM

E2E Training that matches real-world deployment:
1. LSTM maintains hidden state across episode steps
2. When observation arrives: use observation sequence
3. When no observation: use autoregressive prediction

Training Stages:
1. Stage 1: LSTM encoder pre-training (supervised)
2. Stage 2: Behavioral cloning with DAgger
3. Stage 3: SAC fine-tuning (optional, for true RL)
"""


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
    """
    Early stopping based on validation reward.
    
    Stops training when validation reward does not improve for `patience` evaluations.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of evaluations to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = -np.inf
        self.best_step = 0
    
    def check(self, metric: float, current_step: int = 0) -> Tuple[bool, bool]:
        """
        Check if training should stop.
        
        Args:
            metric: Current validation metric (higher is better)
            current_step: Current training step (for logging)
            
        Returns:
            should_stop: True if patience exceeded
            is_best: True if this is a new best
        """
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
        """Reset the stopper for a new training phase."""
        self.counter = 0
        self.best_metric = -np.inf
        self.best_step = 0


class Trainer:
    """
    Trainer that explicitly teaches recovery behavior.
    
    Training Phases:
    1. Stage 1: LSTM encoder pre-training
    2. Stage 2a: Pure BC with noise injection (learn nominal + nearby states)
    3. Stage 2b: DAgger with recovery focus (learn from student mistakes)
    4. Stage 3: SAC fine-tuning (learn optimal recovery through RL)
    """
    
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
        self.target_entropy = -cfg.N_JOINTS
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.ALPHA_LR)
        
        # Buffer
        self.buffer = RecoveryBuffer(cfg.BUFFER_SIZE, cfg.OBS_DIM, cfg.N_JOINTS, self.device)
        
        # Early stopping
        self.stopper = EarlyStopper(
            patience=getattr(cfg, 'EARLY_STOP_PATIENCE', 10),
            min_delta=1.0  # Require at least 1.0 reward improvement
        )
        
        # Training state
        self.global_step = 0
        self._episode_count = 0
        
        # Noise schedule (curriculum)
        self.noise_scale = 0.0
        self.max_noise_scale = 5.0  # Nm
        
    def _get_alpha(self):
        return self.log_alpha.exp()

    def train_stage_1_encoder(self):
        """
        Stage 1: Supervised LSTM encoder pre-training.
        
        Train the LSTM to predict current state from delayed observations.
        Policy layers are frozen during this stage.
        """
        logger.info("=" * 60)
        logger.info(">>> STAGE 1: ENCODER PRE-TRAINING")
        logger.info("=" * 60)
        
        # Freeze policy layers (only train encoder)
        for param in self.actor.backbone.parameters():
            param.requires_grad = False
        for param in self.actor.fc_mean.parameters():
            param.requires_grad = False
        for param in self.actor.fc_log_std.parameters():
            param.requires_grad = False
        
        updates_completed = 0
        COLLECTION_STEPS = 5000
        STAGE1_STEPS = getattr(cfg, 'STAGE1_STEPS', 50000)
        
        while updates_completed < STAGE1_STEPS:
            # Collect data with Teacher (no noise for encoder training)
            logger.info(f"Collecting {COLLECTION_STEPS} steps for encoder training...")
            self.collect_with_noise_injection(
                steps=COLLECTION_STEPS,
                noise_scale=0.0,  # No noise for encoder pre-training
                noise_type='gaussian'
            )
            
            # Train encoder
            logger.info(f"Training encoder...")
            for _ in range(COLLECTION_STEPS):
                if updates_completed >= STAGE1_STEPS:
                    break
                
                batch = self.buffer.sample_with_recovery_weight(cfg.BATCH_SIZE, recovery_weight=1.0)
                obs = batch['obs']
                
                # Extract target history from observation
                target_history = obs[:, -cfg.TARGET_HISTORY_DIM:]
                history_seq = target_history.view(-1, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
                
                # Forward pass through encoder
                _, pred_state, _ = self.shared_encoder(history_seq)
                
                # Get ground truth
                idxs = np.random.randint(0, self.buffer.size, size=cfg.BATCH_SIZE)
                true_states = torch.FloatTensor(self.buffer.true_states[idxs]).to(self.device)
                
                # Normalize targets
                t_q = true_states[:, :7]
                t_qd = true_states[:, 7:]
                t_q_norm = (t_q - torch.tensor(cfg.Q_MEAN, device=self.device)) / torch.tensor(cfg.Q_STD, device=self.device)
                t_qd_norm = (t_qd - torch.tensor(cfg.QD_MEAN, device=self.device)) / torch.tensor(cfg.QD_STD, device=self.device)
                target = torch.cat([t_q_norm, t_qd_norm], dim=1)
                
                # Loss
                loss = F.mse_loss(pred_state, target)
                
                self.enc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.shared_encoder.parameters(), max_norm=1.0)
                self.enc_optimizer.step()
                
                updates_completed += 1
                
                if updates_completed % 1000 == 0:
                    logger.info(f"[Stage 1] Update {updates_completed}/{STAGE1_STEPS} | Loss: {loss.item():.6f}")
                    self.writer.add_scalar("Stage1/EncoderLoss", loss.item(), updates_completed)
        
        # Unfreeze policy layers for Stage 2
        for param in self.actor.parameters():
            param.requires_grad = True
        
        logger.info(">>> STAGE 1 COMPLETE")
        self.save_checkpoint("stage1_complete.pth")
    
    # =========================================================================
    # STAGE 2a: BC with Noise Injection
    # =========================================================================
    
    def collect_with_noise_injection(
        self,
        steps: int,
        noise_scale: float = 2.0,
        noise_type: str = 'gaussian'
    ):
        """
        Collect data with Teacher + Noise.
        
        This exposes the student to states NEAR the optimal trajectory,
        teaching it to recover from small perturbations.
        
        Args:
            steps: Number of steps to collect
            noise_scale: Standard deviation of noise (Nm)
            noise_type: 'gaussian', 'uniform', or 'occasional'
        """
        obs, info = self.env.reset()
        hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
        prev_prediction = torch.zeros(1, 14, device=self.device)
        
        episode_step = 0
        episodes_completed = 0
        
        for i in range(steps):
            episode_step += 1
            has_new_obs = info.get('has_new_obs', True)
            
            # Get Teacher action
            r_q, r_qd = self.env.remote.get_joint_state()
            t_q, t_qd = self.env.leader_hist[-1]
            teacher_action = self.env._compute_teacher_torque(r_q, r_qd, t_q, t_qd)
            
            # Add noise to driving action (but NOT to the label)
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_scale, size=teacher_action.shape)
            elif noise_type == 'uniform':
                noise = np.random.uniform(-noise_scale, noise_scale, size=teacher_action.shape)
            elif noise_type == 'occasional':
                # Add noise only 30% of the time (larger magnitude)
                if np.random.random() < 0.3:
                    noise = np.random.normal(0, noise_scale * 2, size=teacher_action.shape)
                else:
                    noise = np.zeros_like(teacher_action)
            else:
                noise = np.zeros_like(teacher_action)
            
            driving_action = teacher_action + noise
            driving_action = np.clip(driving_action, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
            
            # Get student prediction (for logging, not driving)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, _, pred, next_hidden = self.actor.sample(
                    obs_t, hidden=hidden, prev_prediction=prev_prediction,
                    has_new_obs=has_new_obs, deterministic=True
                )
            
            # Step environment with noisy teacher
            next_obs, reward, terminated, truncated, next_info = self.env.step(driving_action)
            done = terminated or truncated
            
            true_state = next_info['true_state']
            
            # Store: observation, CLEAN teacher label, noisy driving action
            self.buffer.add(
                obs=obs,
                action=driving_action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                teacher_action=teacher_action,  # Clean label!
                true_state=true_state,
                has_new_obs=has_new_obs,
                prev_prediction=prev_prediction.cpu().numpy()[0],
                is_recovery=terminated  # Mark if this led to failure
            )
            
            # Update state
            obs = next_obs
            info = next_info
            hidden = next_hidden
            prev_prediction = pred
            
            if done:
                self._episode_count += 1
                episodes_completed += 1
                
                obs, info = self.env.reset()
                hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
                prev_prediction = torch.zeros(1, 14, device=self.device)
                episode_step = 0
        
        logger.info(
            f"Collected {steps} steps with noise={noise_scale:.1f}Nm, "
            f"{episodes_completed} episodes"
        )

    # =========================================================================
    # STAGE 2b: DAgger with Recovery Focus
    # =========================================================================
    
    def collect_with_recovery_focus(
        self,
        steps: int,
        student_prob: float = 0.5,
        recovery_bonus_steps: int = 50
    ):
        """
        DAgger collection that emphasizes recovery situations.
        
        When student causes an error, we:
        1. Record the error state
        2. Let Teacher demonstrate recovery for next N steps
        3. Mark these as "recovery" samples (can be weighted higher in loss)
        
        Args:
            steps: Number of steps to collect
            student_prob: Probability of student driving
            recovery_bonus_steps: Extra steps to record after error
        """
        obs, info = self.env.reset()
        hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
        prev_prediction = torch.zeros(1, 14, device=self.device)
        
        episode_step = 0
        episodes_completed = 0
        recovery_mode = False
        recovery_steps_remaining = 0
        
        for i in range(steps):
            episode_step += 1
            has_new_obs = info.get('has_new_obs', True)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Decide who drives
            if recovery_mode:
                # During recovery, Teacher ALWAYS drives
                use_student = False
                recovery_steps_remaining -= 1
                if recovery_steps_remaining <= 0:
                    recovery_mode = False
            else:
                use_student = np.random.random() < student_prob
            
            # Get Teacher action (always computed for label)
            r_q, r_qd = self.env.remote.get_joint_state()
            t_q, t_qd = self.env.leader_hist[-1]
            teacher_action = self.env._compute_teacher_torque(r_q, r_qd, t_q, t_qd)
            
            # Get driving action
            if use_student:
                with torch.no_grad():
                    action, _, _, pred, next_hidden = self.actor.sample(
                        obs_t, hidden=hidden, prev_prediction=prev_prediction,
                        has_new_obs=has_new_obs, deterministic=True
                    )
                driving_action = action.cpu().numpy()[0]
            else:
                driving_action = teacher_action
                with torch.no_grad():
                    _, _, _, pred, next_hidden = self.actor.sample(
                        obs_t, hidden=hidden, prev_prediction=prev_prediction,
                        has_new_obs=has_new_obs, deterministic=True
                    )
            
            # Step
            next_obs, reward, terminated, truncated, next_info = self.env.step(driving_action)
            done = terminated or truncated
            
            # Check for error (potential recovery situation)
            tracking_error = next_info.get('tracking_error', 0.0)
            is_recovery_sample = recovery_mode or (tracking_error > 0.1)  # Threshold
            
            # If student caused significant error, switch to recovery mode
            if use_student and tracking_error > 0.05 and not recovery_mode:
                recovery_mode = True
                recovery_steps_remaining = recovery_bonus_steps
                logger.debug(f"Recovery mode activated at step {i}, error={tracking_error:.3f}")
            
            # Store
            self.buffer.add(
                obs=obs,
                action=driving_action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                teacher_action=teacher_action,
                true_state=next_info['true_state'],
                has_new_obs=has_new_obs,
                prev_prediction=prev_prediction.cpu().numpy()[0],
                is_recovery=is_recovery_sample
            )
            
            # Update
            obs = next_obs
            info = next_info
            hidden = next_hidden
            prev_prediction = pred
            
            if done:
                self._episode_count += 1
                episodes_completed += 1
                
                obs, info = self.env.reset()
                hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
                prev_prediction = torch.zeros(1, 14, device=self.device)
                episode_step = 0
                recovery_mode = False
                recovery_steps_remaining = 0
        
        # Log recovery statistics
        recovery_ratio = self.buffer.get_recovery_ratio()
        logger.info(
            f"Collected {steps} steps, {episodes_completed} episodes, "
            f"recovery_ratio={recovery_ratio:.2%}"
        )

    # =========================================================================
    # Training with Recovery Weighting
    # =========================================================================
    
    def train_bc_with_recovery_weighting(self, num_updates: int, recovery_weight: float = 2.0):
        """
        Train BC loss with higher weight on recovery samples.
        
        This makes the model pay more attention to "how to recover"
        rather than "how to follow perfectly".
        """
        total_loss = 0.0
        
        for _ in range(num_updates):
            batch = self.buffer.sample_with_recovery_weight(
                cfg.BATCH_SIZE, 
                recovery_weight=recovery_weight
            )
            
            obs = batch['obs']
            teacher_actions = batch['teacher_actions']
            sample_weights = batch['weights']  # Higher for recovery samples
            
            # Student action
            student_action, _, _, _, _ = self.actor.sample(
                obs, hidden=None, prev_prediction=None,
                has_new_obs=True, deterministic=True
            )
            
            # Weighted BC loss
            scale = self.actor.action_scale
            per_sample_loss = F.mse_loss(
                student_action / scale, 
                teacher_actions / scale,
                reduction='none'
            ).mean(dim=1)
            
            # Apply weights
            weighted_loss = (per_sample_loss * sample_weights).mean()
            
            self.actor_optimizer.zero_grad()
            self.enc_optimizer.zero_grad()
            weighted_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.shared_encoder.parameters(), max_norm=1.0)
            
            self.actor_optimizer.step()
            self.enc_optimizer.step()
            
            total_loss += weighted_loss.item()
        
        return total_loss / num_updates

    # =========================================================================
    # Full Training Pipeline
    # =========================================================================
    
    def train_stage_2_with_recovery(self):
        """
        Improved Stage 2 with explicit recovery learning.
        
        Phase 2a: BC with noise injection (curriculum)
        Phase 2b: DAgger with recovery focus
        """
        logger.info("=" * 60)
        logger.info(">>> STAGE 2: BC WITH RECOVERY LEARNING")
        logger.info("=" * 60)
        
        self.global_step = 0
        COLLECTION_STEPS = 5000
        
        # =====================================================================
        # Phase 2a: BC with Noise Injection (Curriculum)
        # =====================================================================
        logger.info("--- Phase 2a: BC with Noise Injection ---")
        
        PHASE_2A_STEPS = cfg.STAGE2A_TOTAL_STEPS
        
        noise_schedule = [
            (0, 0.5),       # Steps 0-10k: very small noise
            (10000, 1.0),   # Steps 10k-20k: small noise
            (20000, 2.0),   # Steps 20k-30k: medium noise
            (30000, 3.0),   # Steps 30k-40k: larger noise
            (40000, 4.0),   # Steps 40k-50k: significant noise
        ]
        
        while self.global_step < PHASE_2A_STEPS:
            # Get noise scale from schedule
            noise_scale = 0.5
            for threshold, scale in noise_schedule:
                if self.global_step >= threshold:
                    noise_scale = scale
            
            logger.info(f"Step {self.global_step} | Noise injection, scale={noise_scale:.1f}Nm")
            
            self.collect_with_noise_injection(
                steps=COLLECTION_STEPS,
                noise_scale=noise_scale,
                noise_type='gaussian'
            )
            
            self.global_step += COLLECTION_STEPS
            
            # Train
            avg_loss = self.train_bc_with_recovery_weighting(
                num_updates=COLLECTION_STEPS,
                recovery_weight=1.5  # Slight preference for recovery
            )
            
            logger.info(f"[Phase 2a] Avg BC Loss: {avg_loss:.6f}")
            self.writer.add_scalar("Stage2a/BCLoss", avg_loss, self.global_step)
            self.writer.add_scalar("Stage2a/NoiseScale", noise_scale, self.global_step)
            
            # Validation
            if self.global_step % cfg.VAL_FREQ == 0:
                val_reward = self.validate()
                logger.info(f"[Validation] Reward: {val_reward:.2f}")
                self.writer.add_scalar("Validation/Reward", val_reward, self.global_step)
                
                # Early stopping check
                should_stop, is_best = self.stopper.check(val_reward, self.global_step)
                
                if is_best:
                    self.save_checkpoint("best_policy_phase2a.pth")
                    logger.info(">>> New Best Model Saved!")
                
                if should_stop:
                    logger.info(">>> Early Stopping in Phase 2a")
                    break
        
        # Reset stopper for Phase 2b
        self.stopper.reset()
        
        # =====================================================================
        # Phase 2b: DAgger with Recovery Focus
        # =====================================================================
        logger.info("--- Phase 2b: DAgger with Recovery Focus ---")
        
        PHASE_2B_STEPS = cfg.STAGE2_TOTAL_STEPS - PHASE_2A_STEPS
        
        # DAgger schedule (slower ramp)
        while self.global_step < cfg.STAGE2_TOTAL_STEPS:
            # Student probability: 0% -> 30% over phase 2b
            progress = (self.global_step - PHASE_2A_STEPS) / PHASE_2B_STEPS
            student_prob = min(0.3, progress * 0.3)
            
            logger.info(f"Step {self.global_step} | DAgger, p_student={student_prob:.2f}")
            
            self.collect_with_recovery_focus(
                steps=COLLECTION_STEPS,
                student_prob=student_prob,
                recovery_bonus_steps=50
            )
            
            self.global_step += COLLECTION_STEPS
            
            # Train with higher recovery weight
            avg_loss = self.train_bc_with_recovery_weighting(
                num_updates=COLLECTION_STEPS,
                recovery_weight=2.0  # Higher weight for recovery samples
            )
            
            logger.info(f"[Phase 2b] Avg BC Loss: {avg_loss:.6f}")
            self.writer.add_scalar("Stage2b/BCLoss", avg_loss, self.global_step)
            self.writer.add_scalar("Stage2b/StudentProb", student_prob, self.global_step)
            
            # Validation
            if self.global_step % cfg.VAL_FREQ == 0:
                val_reward = self.validate()
                logger.info(f"[Validation] Reward: {val_reward:.2f}")
                self.writer.add_scalar("Validation/Reward", val_reward, self.global_step)
                
                # Early stopping check
                should_stop, is_best = self.stopper.check(val_reward, self.global_step)
                
                if is_best:
                    self.save_checkpoint("best_policy.pth")
                    logger.info(">>> New Best Model Saved!")
                
                if should_stop:
                    logger.info(">>> Early Stopping in Phase 2b")
                    break
        
        logger.info(">>> STAGE 2 COMPLETE")

    # =========================================================================
    # STAGE 3: SAC Fine-tuning
    # =========================================================================

    def train_stage_3_sac(self):
        """
        Stage 3: SAC Fine-tuning (Reinforcement Learning).
        
        In this stage, the agent learns from its own experience to maximize 
        reward (minimize tracking error) using the SAC algorithm.
        """
        logger.info("=" * 60)
        logger.info(">>> STAGE 3: SAC FINE-TUNING")
        logger.info("=" * 60)
        
        # Reset stopper for Stage 3
        self.stopper.reset()
        
        # Hyperparameters
        STAGE3_STEPS = getattr(cfg, 'STAGE3_TOTAL_STEPS', 100000)
        UPDATES_PER_STEP = 1
        LOG_INTERVAL = 1000
        
        # Reset environment for pure RL
        obs, info = self.env.reset()
        hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
        prev_prediction = torch.zeros(1, 14, device=self.device)
        
        # Statistics
        ep_reward = 0
        ep_len = 0
        
        while self.global_step < STAGE3_STEPS:
            self.global_step += 1
            
            # 1. Select Action (Stochastic for exploration)
            has_new_obs = info.get('has_new_obs', True)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Note: We use deterministic=False for exploration in Stage 3
                action, _, _, pred, next_hidden = self.actor.sample(
                    obs_t, hidden=hidden, prev_prediction=prev_prediction,
                    has_new_obs=has_new_obs, deterministic=False
                )
            
            driving_action = action.cpu().numpy()[0]
            
            # 2. Step Environment
            next_obs, reward, terminated, truncated, next_info = self.env.step(driving_action)
            done = terminated or truncated
            
            # 3. Store in Buffer
            # Note: teacher_action is stored but not strictly used for SAC loss, 
            # though we keep it if we want to add auxiliary BC loss.
            true_state = next_info['true_state']
            
            # Get teacher action just for logging/compatibility (optional)
            r_q, r_qd = self.env.remote.get_joint_state()
            t_q, t_qd = self.env.leader_hist[-1]
            teacher_action = self.env._compute_teacher_torque(r_q, r_qd, t_q, t_qd)

            self.buffer.add(
                obs=obs,
                action=driving_action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                teacher_action=teacher_action,
                true_state=true_state,
                has_new_obs=has_new_obs,
                prev_prediction=prev_prediction.cpu().numpy()[0],
                is_recovery=terminated
            )
            
            # Update state
            obs = next_obs
            info = next_info
            hidden = next_hidden
            prev_prediction = pred
            ep_reward += reward
            ep_len += 1
            
            # 4. Update Parameters (SAC Step)
            if self.buffer.current_size > cfg.BATCH_SIZE:
                c_loss, a_loss, alpha_loss, alpha_val = self.update_parameters_sac(UPDATES_PER_STEP)
                
                if self.global_step % LOG_INTERVAL == 0:
                    self.writer.add_scalar("Stage3/CriticLoss", c_loss, self.global_step)
                    self.writer.add_scalar("Stage3/ActorLoss", a_loss, self.global_step)
                    self.writer.add_scalar("Stage3/Alpha", alpha_val, self.global_step)

            # 5. Handle Episode End
            if done:
                logger.info(f"Step {self.global_step} | Episode Reward: {ep_reward:.2f} | Len: {ep_len}")
                self.writer.add_scalar("Stage3/EpReward", ep_reward, self.global_step)
                
                obs, info = self.env.reset()
                hidden = self.shared_encoder.init_hidden(batch_size=1, device=self.device)
                prev_prediction = torch.zeros(1, 14, device=self.device)
                ep_reward = 0
                ep_len = 0
            
            # 6. Validation
            if self.global_step % cfg.VAL_FREQ == 0:
                val_reward = self.validate()
                logger.info(f"[Stage 3 Validation] Reward: {val_reward:.2f}")
                self.writer.add_scalar("Validation/Reward", val_reward, self.global_step)
                
                should_stop, is_best = self.stopper.check(val_reward, self.global_step)
                if is_best:
                    self.save_checkpoint("best_policy_stage3.pth")
                if should_stop:
                    break

        logger.info(">>> STAGE 3 COMPLETE")

    def update_parameters_sac(self, updates: int):
        """
        Standard SAC update loop adapted for the LSTM architecture.
        """
        c_loss_accum, a_loss_accum, alpha_loss_accum = 0, 0, 0
        alpha = self._get_alpha()
        gamma = cfg.GAMMA
        tau = getattr(cfg, 'POLYAK_TAU', 0.005)

        for _ in range(updates):
            # 1. Sample Batch
            # In Stage 3, we usually sample uniformly, or we can keep recovery weighting
            batch = self.buffer.sample_with_recovery_weight(cfg.BATCH_SIZE, recovery_weight=1.0)
            
            obs = batch['obs']
            next_obs = batch['next_obs']
            actions = batch['actions']
            rewards = batch['rewards']
            dones = batch['dones']
            
            # ==============================
            # Critic Update
            # ==============================
            with torch.no_grad():
                # Get next action from target policy
                # Note: We pass has_new_obs=True to rely on the history window in next_obs
                next_action, next_log_prob, _, _, _ = self.actor.sample(
                    next_obs, hidden=None, prev_prediction=None, 
                    has_new_obs=True, deterministic=False
                )
                
                # Target Q-values
                target_q1, target_q2, _ = self.critic_target(
                    next_obs, next_action, hidden=None, prev_prediction=None, has_new_obs=True
                )
                min_target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                target_q = rewards + (1 - dones) * gamma * min_target_q

            # Current Q-values
            current_q1, current_q2, _ = self.critic(
                obs, actions, hidden=None, prev_prediction=None, has_new_obs=True
            )
            
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            self.enc_optimizer.zero_grad() # Encoder gradients flow from Critic too
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.shared_encoder.parameters(), 1.0)
            self.critic_optimizer.step()
            self.enc_optimizer.step()

            # ==============================
            # Actor Update
            # ==============================
            # Sample action from current policy
            new_action, log_prob, _, _, _ = self.actor.sample(
                obs, hidden=None, prev_prediction=None, 
                has_new_obs=True, deterministic=False
            )
            
            # Get Q-values for new action
            # Note: We detach encoder here to prevent Actor update from messing up representation 
            # trained by Critic/Supervised loss, or we can allow it. 
            # Usually standard SAC allows it.
            q1_pi, q2_pi, _ = self.critic(
                obs, new_action, hidden=None, prev_prediction=None, has_new_obs=True
            )
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            actor_loss = (alpha * log_prob - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # ==============================
            # Alpha Update (Automatic Entropy Tuning)
            # ==============================
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self._get_alpha()

            # ==============================
            # Polyak Averaging (Target Networks)
            # ==============================
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.mul_(1 - tau)
                    target_param.data.add_(tau * param.data)
            
            c_loss_accum += critic_loss.item()
            a_loss_accum += actor_loss.item()
            alpha_loss_accum += alpha_loss.item()

        return (
            c_loss_accum / updates, 
            a_loss_accum / updates, 
            alpha_loss_accum / updates, 
            alpha.item()
        )
    
    def validate(self) -> float:
        """Run validation."""
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
                obs = next_obs
                info = next_info
                hidden = next_hidden
                prev_prediction = pred
                
                if done:
                    break
            
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


class RecoveryBuffer:
    """
    Replay buffer that tracks recovery samples separately.
    """
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.teacher_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.true_states = np.zeros((capacity, 14), dtype=np.float32)
        self.has_new_obs = np.zeros((capacity, 1), dtype=np.float32)
        self.prev_predictions = np.zeros((capacity, 14), dtype=np.float32)
        self.is_recovery = np.zeros((capacity, 1), dtype=np.float32)  # NEW
    
    def add(
        self,
        obs, action, reward, next_obs, done,
        teacher_action, true_state, has_new_obs, prev_prediction,
        is_recovery: bool = False
    ):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.teacher_actions[self.ptr] = teacher_action
        self.true_states[self.ptr] = true_state
        self.has_new_obs[self.ptr] = float(has_new_obs)
        self.prev_predictions[self.ptr] = prev_prediction
        self.is_recovery[self.ptr] = float(is_recovery)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample_with_recovery_weight(
        self, 
        batch_size: int, 
        recovery_weight: float = 2.0
    ) -> Dict[str, torch.Tensor]:
        """
        Sample with higher probability for recovery samples.
        """
        # Compute sampling weights
        weights = np.ones(self.size)
        recovery_mask = self.is_recovery[:self.size, 0] > 0.5
        weights[recovery_mask] = recovery_weight
        weights = weights / weights.sum()
        
        # Weighted sampling
        idxs = np.random.choice(self.size, size=batch_size, p=weights)
        
        # Compute loss weights (inverse of sampling probability for unbiased gradients)
        # Or just use recovery flag as weight
        sample_weights = np.ones(batch_size)
        sample_weights[self.is_recovery[idxs, 0] > 0.5] = recovery_weight
        
        return {
            'obs': torch.FloatTensor(self.obs[idxs]).to(self.device),
            'teacher_actions': torch.FloatTensor(self.teacher_actions[idxs]).to(self.device),
            'weights': torch.FloatTensor(sample_weights).to(self.device),
            'is_recovery': torch.FloatTensor(self.is_recovery[idxs]).to(self.device),
        }
    
    def get_recovery_ratio(self) -> float:
        """Get ratio of recovery samples in buffer."""
        if self.size == 0:
            return 0.0
        return self.is_recovery[:self.size].mean()
    
    @property
    def current_size(self):
        return self.size