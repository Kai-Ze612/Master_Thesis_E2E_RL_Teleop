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
from E2E_Teleoperation.E2E_RL.sac_policy_network import LSTM, JointActor, JointCritic
from E2E_Teleoperation.E2E_RL.sac_training_algorithm import SACAlgorithm

class ReplayBuffer:
    """
    Standard Replay Buffer that also stores 'teacher_action' and 'true_state'
    needed for the specific auxiliary losses in Stage 1 & 2.
    """
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Standard RL
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # Aux Data (for BC and Encoder training)
        self.teacher_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.true_states = np.zeros((capacity, 14), dtype=np.float32) # 7 pos + 7 vel

    def add(self, obs, action, reward, next_obs, done, teacher_action, true_state):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        
        self.teacher_actions[self.ptr] = teacher_action
        self.true_states[self.ptr] = true_state
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.FloatTensor(self.obs[idxs]).to(self.device),
            'next_obs': torch.FloatTensor(self.next_obs[idxs]).to(self.device),
            'actions': torch.FloatTensor(self.actions[idxs]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[idxs]).to(self.device),
            'dones': torch.FloatTensor(self.dones[idxs]).to(self.device),
            'teacher_actions': torch.FloatTensor(self.teacher_actions[idxs]).to(self.device),
            'true_states': torch.FloatTensor(self.true_states[idxs]).to(self.device)
        }

class UnifiedTrainer:
    def __init__(self, env, output_dir):
        self.env = env
        self.output_dir = Path(output_dir)
        self.log_file = self.output_dir / "training_log.txt"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize Networks
        self.encoder = ContinuousLSTMEncoder().to(self.device)
        self.actor = JointActor(self.encoder).to(self.device)
        self.critic = JointCritic(self.encoder).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 2. Buffer
        self.buffer = ReplayBuffer(cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.OBS_DIM, 7, self.device)
        
        # 3. Alpha (Entropy)
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
        """
        # Get Predicted State for logging (Forward pass without gradients)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            # FIX: Unpack the 3rd element (pred_state), not the 2nd
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

    def _collect_data(self, steps, random=False):
        """Helper to fill buffer before training starts"""
        self._log(f">> Collecting {steps} steps of data...")
        obs, info = self.env.reset()
        for _ in range(steps):
            if random:
                action = self.env.action_space.sample()
            else:
                action = info['teacher_action']
            
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            # Store teacher action and true state for supervised losses
            self.buffer.add(obs, action, reward, next_obs, terminated, 
                          next_info['teacher_action'], next_info['true_state_vector'])
            
            obs = next_obs
            info = next_info
            if terminated or truncated:
                obs, info = self.env.reset()

    # =========================================================
    # STAGE 1: ENCODER PRE-TRAINING
    # =========================================================
    def train_stage1(self):
        """Train Encoder Only using True State Labels"""
        self._log(">>> STAGE 1: Encoder Pre-training")
        
        # 1. Collect Data (Teacher Policy)
        self._collect_data(10000, random=False)
        
        # 2. Train
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        
        for step in range(cfg.TRAIN.STAGE1_STEPS):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            
            # FIX: Correct unpacking (pred_state is 3rd return value)
            # Returns: mu, log_std, pred_state, next_hidden, feat
            _, _, pred_state, _, _ = self.actor.forward(batch['obs'])
            
            # Loss: MSE between Predicted State (14) and True State (14)
            loss = F.mse_loss(pred_state, batch['true_states'])
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            if step % 1000 == 0:
                self._log(f"Stage 1 | Step {step} | Pred Loss: {loss.item():.5f}")
        
        torch.save(self.encoder.state_dict(), self.output_dir / "stage1_final.pth")

    # =========================================================
    # STAGE 2: POLICY BC PRE-TRAINING
    # =========================================================
    def train_stage2_bc(self):
        """Train Policy Only (Behavioral Cloning)"""
        self._log(">>> STAGE 2: BC Pre-training")
        
        # Freeze Encoder (Assume it's good from Stage 1)
        for p in self.encoder.parameters(): p.requires_grad = False
        
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        for step in range(cfg.TRAIN.STAGE2_STEPS):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            
            # Forward Actor (mu is 1st return value)
            mu, _, _, _, _ = self.actor.forward(batch['obs'])
            pred_action = torch.tanh(mu) * self.actor.scale.to(self.device)
            
            # Loss: MSE between Actor Action and Teacher Action
            loss = F.mse_loss(pred_action, batch['teacher_actions'])
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            if step % 1000 == 0:
                self._log(f"Stage 2 | Step {step} | BC Loss: {loss.item():.5f}")
        
        # Unfreeze for Stage 3
        for p in self.encoder.parameters(): p.requires_grad = True
        torch.save(self.actor.state_dict(), self.output_dir / "stage2_final.pth")

    # =========================================================
    # STAGE 3: END-TO-END SAC
    # =========================================================
    def train_stage3_sac(self):
        self._log("\n>>> STAGE 3: End-to-End SAC Fine-tuning")
        
        # Optimizers (Differentiated LRs)
        opt_actor = torch.optim.Adam([
            {'params': self.actor.net.parameters(), 'lr': 1e-4}, 
            {'params': self.encoder.parameters(), 'lr': 1e-5}
        ], weight_decay=1e-5)
        
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