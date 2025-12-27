"""
E2E_Teleoperation/E2E_RL/sac_training_algorithm.py
"""
import numpy as np
import torch
import torch.nn.functional as F
import E2E_Teleoperation.config.robot_config as cfg

class SACAlgorithm:
    def __init__(self, actor, critic, critic_target, actor_opt, critic_opt, alpha_opt, log_alpha):
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optimizer = actor_opt
        self.critic_optimizer = critic_opt
        self.alpha_optimizer = alpha_opt
        self.log_alpha = log_alpha
        self.target_entropy = -float(cfg.ROBOT.N_JOINTS) * cfg.SAC.TARGET_ENTROPY_RATIO
        self.gamma = cfg.TRAIN.GAMMA
        self.tau = cfg.SAC.TARGET_TAU
        self.device = actor.scale.device

    def update(self, batch):
        obs = batch['obs']
        action = batch['actions']
        reward = batch['rewards']
        next_obs = batch['next_obs']
        not_done = 1.0 - batch['dones']
        
        # Ground Truths for Aux Losses
        true_state = batch['true_states']     # For Encoder Loss
        teacher_action = batch['teacher_actions'] # For BC Loss

        # -------------------------
        # 1. Critic Update
        # -------------------------
        with torch.no_grad():
            next_action, next_log_prob, _, _, next_feat = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_feat, next_action)
            target_q = torch.min(target_q1, target_q2)
            alpha = self.log_alpha.exp()
            target_value = reward + not_done * self.gamma * (target_q - alpha * next_log_prob)

        # Get current features (gradients flow to encoder here)
        _, _, _, _, curr_feat = self.actor.forward(obs)
        current_q1, current_q2 = self.critic(curr_feat, action)
        
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------------
        # 2. Actor & Encoder Update
        # -------------------------
        new_action, log_prob, pred_state, _, feat = self.actor.sample(obs)
        q1_new, q2_new = self.critic(feat, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        alpha = self.log_alpha.exp()
        sac_loss = (alpha * log_prob - q_new).mean()
        
        # Aux 1: Prediction Loss (MSE between Predicted State and True State)
        pred_loss = F.mse_loss(pred_state, true_state)
        
        # Aux 2: BC Regularization (MSE between Actor and Teacher)
        scale = self.actor.scale
        bc_loss = F.mse_loss(new_action/scale, teacher_action/scale)
        
        total_actor_loss = sac_loss + 1.0 * pred_loss + cfg.SAC.BC_MIN_WEIGHT * bc_loss
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()

        # -------------------------
        # 3. Alpha Update
        # -------------------------
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # -------------------------
        # 4. Soft Update
        # -------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": sac_loss.item(),
            "pred_loss": pred_loss.item(),
            "bc_loss": bc_loss.item(),
            "alpha": alpha.item()
        }