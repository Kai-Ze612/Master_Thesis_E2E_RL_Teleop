"""
E2E_Teleoperation/E2E_RL/unified_trainer.py

PURE BEHAVIORAL CLONING - No SAC at all.
Just imitate the teacher exactly.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import copy
from pathlib import Path

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.sac_policy_network import ContinuousLSTMEncoder, JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.sac_training_algorithm import RecoveryBuffer


class UnifiedTrainer:
    def __init__(self, env, output_dir):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_path = self.output_dir / "training_log.txt"
        if not self.log_path.exists():
            with open(self.log_path, "w") as f:
                f.write(f"Training started in: {self.output_dir}\n")
                f.write("="*60 + "\n")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Models
        self.encoder = ContinuousLSTMEncoder().to(self.device)
        self.actor = JointActor(self.encoder).to(self.device)
        
        # No critic needed for pure BC
        self.critic = JointCritic(self.encoder).to(self.device)  # Keep for checkpoint compatibility
        self.critic_target = copy.deepcopy(self.critic)
        
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # 2. Optimizers - Only actor needed
        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=cfg.TRAIN.ENCODER_LR)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-4)  # Low LR for stability
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.TRAIN.CRITIC_LR)
        
        # 3. Buffer
        self.buffer = RecoveryBuffer(cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.OBS_DIM, 7, self.device)
        
        # 4. Training state
        self.total_updates = 0
        self.best_avg_reward = -np.inf

    def log_message(self, message):
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def log_status(self, step, loss_info, info, stage_name, ep_reward, avg_reward):
        summary = (f"[{stage_name} Step {step}] "
                   f"R: {ep_reward:.1f} | "
                   f"AvgR: {avg_reward:.1f} | "
                   f"BC_Loss: {loss_info['bc_loss']:.4f} | "
                   f"Trk: {info['tracking_error']:.3f}")

        with open(self.log_path, "a") as f:
            f.write(summary + "\n")
        print(summary)
        sys.stdout.flush()

    def _freeze_encoder(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.log_message(">> LSTM Encoder is now FROZEN.")

    # =========================================================
    # STAGE 3: PURE BEHAVIORAL CLONING
    # =========================================================
    def train_stage3_sac(self):
        """Pure BC - no SAC components"""
        self.log_message("\n>>> STAGE 3: PURE BEHAVIORAL CLONING")
        self.log_message("    No SAC, no critic, no entropy - just imitate teacher")
        
        # 1. Freeze Encoder
        self._freeze_encoder()
        
        # 2. Initialize Policy with low variance (nearly deterministic)
        with torch.no_grad():
            self.actor.log_std.bias.fill_(-5.0)  # Very low variance
            self.actor.log_std.weight.fill_(0.001)
        self.log_message(">> Policy initialized with very low variance (nearly deterministic).")

        # 3. Fill buffer with teacher demonstrations
        self.log_message(f">> Collecting teacher demonstrations...")
        self._collect_teacher_demos(20000)
        self.log_message(f">> Collected {self.buffer.current_size} transitions.")

        # 4. Main Training Loop
        self.log_message(">> Starting Pure BC Training Loop...")
        obs, info = self.env.reset()
        
        current_ep_reward = 0.0
        last_ep_reward = 0.0
        episode_rewards = []
        loss_info = {'bc_loss': 0.0}

        for step in range(cfg.TRAIN.STAGE3_STEPS):
            # Use DETERMINISTIC action (mean, no sampling)
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                # Get mean action directly, no sampling
                action, _, _ = self.actor.get_action_deterministic(obs_t)
                action = action[0].cpu().numpy()
            
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            current_ep_reward += reward
            
            teacher_tau = info.get('teacher_action', np.zeros(7))
            
            # Add to buffer (for online learning)
            self.buffer.add(
                obs, action, reward, next_obs, terminated, 
                teacher_tau, next_info['true_state_vector'], terminated
            )
            
            # Train with pure BC
            if self.buffer.current_size > cfg.TRAIN.BATCH_SIZE:
                loss_info = self._update_bc_only()

            # Logging
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            
            if step % cfg.TRAIN.LOG_FREQ == 0:
                self.log_status(step, loss_info, info, "BC", last_ep_reward, avg_reward)
            
            # Save best model
            if avg_reward > self.best_avg_reward and len(episode_rewards) >= 10:
                self.best_avg_reward = avg_reward
                save_path = self.output_dir / "best_policy_stage3.pth"
                torch.save(self.actor.state_dict(), str(save_path))
                self.log_message(f">> New best model! Avg Reward: {avg_reward:.2f}")
            
            obs = next_obs
            info = next_info
            
            if terminated or truncated:
                last_ep_reward = current_ep_reward
                episode_rewards.append(current_ep_reward)
                self.log_message(f" >>> EPISODE {len(episode_rewards)} | "
                               f"REWARD: {last_ep_reward:.2f} | AVG(20): {avg_reward:.2f} <<<")
                current_ep_reward = 0.0
                obs, info = self.env.reset()

        # Save Final
        save_path = self.output_dir / "final_policy_stage3.pth"
        torch.save(self.actor.state_dict(), str(save_path))
        self.log_message(f"Saved final checkpoint to: {save_path}")

    # =========================================================
    # COLLECT TEACHER DEMONSTRATIONS
    # =========================================================
    def _collect_teacher_demos(self, num_steps):
        """Run teacher policy and collect demonstrations"""
        obs, info = self.env.reset()
        
        for i in range(num_steps):
            # Use teacher action directly
            action = info['teacher_action']
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            self.buffer.add(
                obs, action, reward, next_obs, terminated, 
                action, next_info['true_state_vector'], False
            )
            
            obs = next_obs
            info = next_info
            
            if terminated or truncated:
                obs, info = self.env.reset()
                
            if i % 5000 == 0:
                print(f"   Collecting demos: {i}/{num_steps}")

    # =========================================================
    # PURE BC UPDATE
    # =========================================================
    def _update_bc_only(self):
        """
        Pure behavioral cloning - minimize MSE between actor output and teacher action.
        No SAC, no critic, no entropy.
        """
        batch = self.buffer.sample_with_recovery_weight(cfg.TRAIN.BATCH_SIZE, recovery_weight=1.0)
        
        # Get actor's mean action (deterministic)
        mu, log_std, _, _ = self.actor.forward(batch['obs'])
        mean_action = torch.tanh(mu) * self.actor.scale.to(self.device)
        
        # BC Loss: MSE between actor output and teacher action
        # Normalize by torque scale to balance joint contributions
        torque_scale = self.actor.scale.to(self.device)
        norm_actor = mean_action / torque_scale
        norm_teacher = batch['teacher_actions'] / torque_scale
        
        bc_loss = F.mse_loss(norm_actor, norm_teacher)
        
        self.opt_actor.zero_grad()
        bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_actor.step()
        
        self.total_updates += 1
        
        return {'bc_loss': bc_loss.item()}

    def _soft_update_target(self):
        pass  # Not needed for pure BC

    def train_stage1(self):
        self.log_message("Skipping Stage 1 code for brevity")

    def train_stage2_bc(self):
        self.log_message("Skipping Stage 2 code for brevity")