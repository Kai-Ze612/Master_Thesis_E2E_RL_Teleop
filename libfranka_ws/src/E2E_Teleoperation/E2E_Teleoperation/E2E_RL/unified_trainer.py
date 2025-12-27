"""
E2E_Teleoperation/E2E_RL/unified_trainer.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import sys
from pathlib import Path

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.sac_policy_network import ContinuousLSTMEncoder, JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.sac_training_algorithm import RecoveryBuffer, SACAlgorithm

class UnifiedTrainer:
    def __init__(self, env, output_dir):
        self.env = env
        self.output_dir = Path(output_dir)
        self.log_file = self.output_dir / "training_log.txt"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Networks
        self.encoder = ContinuousLSTMEncoder().to(self.device)
        self.actor = JointActor(self.encoder).to(self.device)
        self.critic = JointCritic(self.encoder).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Buffer
        self.buffer = RecoveryBuffer(cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.OBS_DIM, 7, self.device)
        
        # Alpha (Entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)

    def _log(self, msg, print_to_console=True):
        """Dual logging to file and console"""
        if print_to_console:
            print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def _log_debug_info(self, step, obs, action, info, metrics=None):
        """
        Specific detailed logging for debugging every 100 steps
        1. True q (Leader)
        2. Predicted q (Internal State)
        3. Remote q (Follower)
        4. RL Output Tau
        5. Losses
        """
        # Get Predicted State for logging (Forward pass without gradients)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _, _, pred_state, _, _ = self.actor.forward(obs_t)
            pred_q = pred_state[0, :7].cpu().numpy()
        
        true_q = info['true_q']
        remote_q = info['remote_q']
        
        log_str = (
            f"\n[DEBUG Step {step}]\n"
            f"--------------------------------------------------\n"
            f"1. True q (Leader):   {np.array2string(true_q, precision=3, suppress_small=True)}\n"
            f"2. Pred q (Encoder):  {np.array2string(pred_q, precision=3, suppress_small=True)}\n"
            f"3. Remote q (Follow): {np.array2string(remote_q, precision=3, suppress_small=True)}\n"
            f"4. RL Torque:         {np.array2string(action, precision=3, suppress_small=True)}\n"
        )
        
        if metrics:
            log_str += (
                f"5. Actor Loss:        {metrics.get('actor_loss', 0.0):.4f}\n"
                f"6. Critic Loss:       {metrics.get('critic_loss', 0.0):.4f}\n"
                f"7. BC Loss:           {metrics.get('bc_loss', 0.0):.4f}\n"
                f"8. Pred Loss:         {metrics.get('pred_loss', 0.0):.4f}\n"
            )
        
        log_str += "--------------------------------------------------"
        self._log(log_str)

    def train_stage3_sac(self):
        self._log("\n>>> STAGE 3: End-to-End SAC Fine-tuning")
        
        # Optimizers
        opt_actor = torch.optim.Adam([
            {'params': self.actor.net.parameters(), 'lr': 1e-4}, 
            {'params': self.encoder.parameters(), 'lr': 1e-5}
        ], weight_decay=1e-5) # Added weight decay for stability
        
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        sac = SACAlgorithm(self.actor, self.critic, self.critic_target, 
                           opt_actor, opt_critic, self.opt_alpha, self.log_alpha)
        
        obs, info = self.env.reset()
        ep_reward = 0
        ep_steps = 0
        
        for step in range(cfg.TRAIN.STAGE3_STEPS):
            
            # 1. Action
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, _, _, _, _ = self.actor.sample(obs_t)
                action = action[0].cpu().numpy()
            
            # 2. Step
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            ep_reward += reward
            ep_steps += 1
            
            # 3. Buffer
            self.buffer.add(obs, action, reward, next_obs, terminated,
                          next_info['teacher_action'], next_info['true_state_vector'])
            
            obs = next_obs
            info = next_info
            
            # 4. Train
            metrics = {}
            if self.buffer.size > cfg.TRAIN.BATCH_SIZE:
                metrics = sac.update(self.buffer.sample(cfg.TRAIN.BATCH_SIZE))

            # 5. Logging
            # A. Debug Log (Every 100 steps)
            if step % 100 == 0:
                self._log_debug_info(step, obs, action, info, metrics)
                
            # B. Episode Log (Only on termination)
            if terminated or truncated:
                stop_reason = "Max Steps" if truncated else "EARLY STOP (Error Divergence)"
                self._log(f"Step {step} | Episode Finish | Reward: {ep_reward:.1f} | Reason: {stop_reason}")
                
                obs, info = self.env.reset()
                ep_reward = 0
                ep_steps = 0
                
            # 6. Checkpointing
            if step % 10000 == 0:
                torch.save(self.actor.state_dict(), self.output_dir / f"stage3_ckpt_{step}.pth")

    def train_stage1(self):
        """Train Encoder Only"""
        self._log(">>> STAGE 1: Encoder Pre-training")
        # (Simplified for brevity, but follows same logging pattern)
        # ... logic to fill buffer ...
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        for step in range(cfg.TRAIN.STAGE1_STEPS):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            with torch.no_grad():
                # We extract true state from batch
                true_state = batch['true_states']
            
            # Forward only encoder part via actor wrapper or direct
            _, _, pred_state, _, _ = self.actor.forward(batch['obs'])
            loss = F.mse_loss(pred_state, true_state)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            if step % 1000 == 0:
                self._log(f"Stage 1 | Step {step} | Pred Loss: {loss.item():.5f}")
        
        torch.save(self.encoder.state_dict(), self.output_dir / "stage1_final.pth")

    def train_stage2_bc(self):
        """Train Policy Only (BC)"""
        self._log(">>> STAGE 2: BC Pre-training")
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        for step in range(cfg.TRAIN.STAGE2_STEPS):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            mu, _, _, _, _ = self.actor.forward(batch['obs'])
            pred_action = torch.tanh(mu) * self.actor.scale.to(self.device)
            loss = F.mse_loss(pred_action, batch['teacher_actions'])
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            if step % 1000 == 0:
                self._log(f"Stage 2 | Step {step} | BC Loss: {loss.item():.5f}")
                
        torch.save(self.actor.state_dict(), self.output_dir / "stage2_final.pth")