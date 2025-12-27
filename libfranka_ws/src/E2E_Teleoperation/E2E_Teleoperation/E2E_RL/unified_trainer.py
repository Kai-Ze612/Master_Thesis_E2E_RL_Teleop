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
    def __init__(self, capacity, obs_dim, action_dim, device):
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
    def __init__(self, env, output_dir, is_vector_env=False):
        self.env = env
        self.output_dir = Path(output_dir)
        self.log_file = self.output_dir / "training_log.txt"
        self.is_vector_env = is_vector_env  # Store the flag
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize Networks
        self.encoder = LSTM().to(self.device)
        self.actor = JointActor(self.encoder).to(self.device)
        self.critic = JointCritic(self.encoder).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 2. Buffer
        self.buffer = ReplayBuffer(cfg.TRAIN.BUFFER_SIZE, cfg.ROBOT.OBS_DIM, 7, self.device)
        
        # 3. Alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.TRAIN.ALPHA_LR)

    def _log(self, msg, print_to_console=True):
        if print_to_console: print(msg)
        with open(self.log_file, "a") as f: f.write(msg + "\n")

    def _log_debug_info(self, step, obs, action, info, metrics=None):
        # Handle Vector Env for Logging (Just take the first env)
        if self.is_vector_env:
            obs_single = obs[0]
            action_single = action[0]
            # Info in vector env is a dictionary of arrays
            true_q = info['true_q'][0]
            remote_q = info['remote_q'][0]
        else:
            obs_single = obs
            action_single = action
            true_q = info['true_q']
            remote_q = info['remote_q']

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_single).unsqueeze(0).to(self.device)
            _, _, pred_state, _, _ = self.actor.forward(obs_t)
            pred_q = pred_state[0, :7].cpu().numpy()
        
        log_str = (
            f"\n[DEBUG Step {step}]\n"
            f"--------------------------------------------------\n"
            f"1. True q (Leader):   {np.array2string(true_q, precision=3, suppress_small=True)}\n"
            f"2. Pred q (Encoder):  {np.array2string(pred_q, precision=3, suppress_small=True)}\n"
            f"3. Remote q (Follow): {np.array2string(remote_q, precision=3, suppress_small=True)}\n"
            f"4. RL Torque:         {np.array2string(action_single, precision=3, suppress_small=True)}\n"
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

    def _add_to_buffer(self, obs, action, reward, next_obs, terminated, truncated, info):
        """Helper to handle both single and vector env data adding"""
        if self.is_vector_env:
            num_envs = len(obs)
            for i in range(num_envs):
                done = terminated[i] or truncated[i]
                self.buffer.add(
                    obs[i], action[i], reward[i], next_obs[i], done,
                    info['teacher_action'][i], info['true_state_vector'][i]
                )
        else:
            done = terminated or truncated
            self.buffer.add(
                obs, action, reward, next_obs, done,
                info['teacher_action'], info['true_state_vector']
            )

    def _collect_data(self, steps, random=False):
        self._log(f">> Collecting {steps} steps of data...")
        obs, info = self.env.reset()
        for _ in range(steps):
            if random: 
                if self.is_vector_env:
                    action = self.env.action_space.sample() # This samples batch in vector env
                else:
                    action = self.env.action_space.sample()
            else: 
                action = info['teacher_action']
            
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            self._add_to_buffer(obs, action, reward, next_obs, terminated, truncated, next_info)
            
            obs = next_obs
            info = next_info
            
            # Reset handling for single env (Vector env auto-resets)
            if not self.is_vector_env and (terminated or truncated):
                obs, info = self.env.reset()

    def train_stage1(self):
        """Train Encoder Only"""
        self._log(">>> STAGE 1: Encoder Pre-training")
        self._collect_data(10000, random=False)
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        best_loss = float('inf')

        for step in range(cfg.TRAIN.STAGE1_STEPS):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            _, _, pred_state, _, _ = self.actor.forward(batch['obs'])
            loss = F.mse_loss(pred_state, batch['true_states'])
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            if step % 100 == 0:
                cur_loss = loss.item()
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    torch.save(self.encoder.state_dict(), self.output_dir / "stage1_best.pth")
                self._log(f"Stage 1 | Step {step} | Loss: {cur_loss:.5f} | Best: {best_loss:.5f}")
        
        torch.save(self.encoder.state_dict(), self.output_dir / "stage1_final.pth")

    def train_stage2_bc(self):
        """Train Policy Only (BC)"""
        self._log(">>> STAGE 2: BC Pre-training")
        for p in self.encoder.parameters(): p.requires_grad = False
        
        optimizer = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad], 
            lr=1e-4
        )
        best_loss = float('inf')

        for step in range(cfg.TRAIN.STAGE2_STEPS):
            batch = self.buffer.sample(cfg.TRAIN.BATCH_SIZE)
            mu, _, _, _, _ = self.actor.forward(batch['obs'])
            pred_action = torch.tanh(mu) * self.actor.scale.to(self.device)
            
            # Normalized BC Loss
            scale = self.actor.scale.to(self.device)
            loss = F.mse_loss(pred_action/scale, batch['teacher_actions']/scale)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            if step % 100 == 0:
                cur_loss = loss.item()
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    torch.save(self.actor.state_dict(), self.output_dir / "stage2_best.pth")
                self._log(f"Stage 2 | Step {step} | BC Loss: {cur_loss:.5f} | Best: {best_loss:.5f}")
        
        for p in self.encoder.parameters(): p.requires_grad = True
        torch.save(self.actor.state_dict(), self.output_dir / "stage2_final.pth")

    def train_stage3_sac(self):
        self._log("\n>>> STAGE 3: End-to-End SAC Fine-tuning")
        
        opt_actor = torch.optim.Adam([
            {'params': self.actor.net.parameters(), 'lr': 1e-4}, 
            {'params': self.encoder.parameters(), 'lr': 1e-5}
        ], weight_decay=1e-5)
        
        critic_params = [p for n, p in self.critic.named_parameters() if 'encoder' not in n]
        opt_critic = torch.optim.Adam(critic_params, lr=3e-4)
        
        sac = SACAlgorithm(self.actor, self.critic, self.critic_target, 
                           opt_actor, opt_critic, self.opt_alpha, self.log_alpha)
        
        obs, info = self.env.reset()
        ep_reward = 0
        
        # Track best reward for saving best model
        best_avg_reward = -float('inf')
        recent_rewards = []
        
        for step in range(cfg.TRAIN.STAGE3_STEPS):
            with torch.no_grad():
                # Handle Vector Obs dimensions
                if self.is_vector_env:
                    obs_t = torch.FloatTensor(obs).to(self.device)
                else:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                action, _, _, _, _ = self.actor.sample(obs_t)
                action = action.cpu().numpy()
                if not self.is_vector_env:
                    action = action[0]

            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            # For logging reward, we sum up if vectorized (just to have a number) 
            # or average. Usually we log when an episode terminates.
            if self.is_vector_env:
                # Add mean reward of the batch to tracker (approximate)
                ep_reward += np.mean(reward)
            else:
                ep_reward += reward

            self._add_to_buffer(obs, action, reward, next_obs, terminated, truncated, next_info)
            
            obs = next_obs
            info = next_info
            
            if self.buffer.size > cfg.TRAIN.BATCH_SIZE:
                metrics = sac.update(self.buffer.sample(cfg.TRAIN.BATCH_SIZE))
                if step % 100 == 0: 
                    self._log_debug_info(step, obs, action, info, metrics)

            # Check termination
            # In VectorEnv, 'terminated' is an array. We log if ANY env terminates.
            if self.is_vector_env:
                if np.any(terminated) or np.any(truncated):
                    # For vector env, accurately tracking episode reward requires an array tracker.
                    # Simplified logging here:
                    self._log(f"Step {step} | Vector Batch Terminated")
                    
                    # NOTE: VectorEnv Auto-resets specific environments, no manual reset needed.
            else:
                if terminated or truncated:
                    stop_reason = "Max Steps" if truncated else "EARLY STOP"
                    self._log(f"Step {step} | Episode Finish | Reward: {ep_reward:.1f} | {stop_reason}")
                    
                    recent_rewards.append(ep_reward)
                    if len(recent_rewards) > 10: recent_rewards.pop(0)
                    avg_rew = np.mean(recent_rewards)
                    
                    if avg_rew > best_avg_reward:
                        best_avg_reward = avg_rew
                        torch.save(self.actor.state_dict(), self.output_dir / "stage3_best.pth")

                    obs, info = self.env.reset()
                    ep_reward = 0
                
            if step % 10000 == 0:
                torch.save(self.actor.state_dict(), self.output_dir / f"stage3_ckpt_{step}.pth")