"""
End-to-End SAC Training Algorithm (Autoregressive LSTM + SAC)

Features:
1. LSTM Estimator: predicts true local robot states
2. Soft Actor-Critic (SAC) for policy learning
3. Autoregressive action generation for handling input delays

End-to-End:
1. Optimize LSTM (Supervised Learning) and SAC (Reinforcement Learning) jointly
2. Data Patching: Updates stale predictions in the Replay Buffer with fresh LSTM output.
3. Gradient Detachment: LSTM learns ONLY from physics loss; SAC learns ONLY from reward.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple
from copy import deepcopy
from collections import deque

from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

from End_to_End_RL_In_Teleoperation.agent.sac_policy_network import Actor, Critic, StateEstimator
from End_to_End_RL_In_Teleoperation.agent.training_env import TeleoperationEnvWithDelay
from End_to_End_RL_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from End_to_End_RL_In_Teleoperation.agent.local_robot_simulator import TrajectoryType

import End_to_End_RL_In_Teleoperation.config.robot_config as cfg

logger = logging.getLogger(__name__)

class ReplayBuffer:
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.obs_dim = cfg.OBS_DIM
        self.action_dim = cfg.N_JOINTS
        self.seq_len = cfg.RNN_SEQUENCE_LENGTH
        self.seq_dim = cfg.ESTIMATOR_STATE_DIM

        # RL Transitions
        self.remote_states = np.zeros((buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_remote_states = np.zeros((buffer_size, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        
        # End-to-End Specific: Raw inputs for LSTM regeneration
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.seq_dim), dtype=np.float32)
        self.true_targets = np.zeros((buffer_size, 14), dtype=np.float32) # q + qd (Ground Truth)
        self.next_delayed_sequences = np.zeros((buffer_size, self.seq_len, self.seq_dim), dtype=np.float32)
        
    def add(self, remote_state, action, reward, next_remote_state, done, 
            delayed_seq, true_target, next_delayed_seq):
        """Add a transition with LSTM-specific data."""
        self.remote_states[self.ptr] = remote_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_remote_states[self.ptr] = next_remote_state
        self.dones[self.ptr] = done
        
        self.delayed_sequences[self.ptr] = delayed_seq
        self.true_targets[self.ptr] = true_target
        self.next_delayed_sequences[self.ptr] = next_delayed_seq
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'remote_states': torch.tensor(self.remote_states[indices], device=self.device),
            'actions': torch.tensor(self.actions[indices], device=self.device),
            'rewards': torch.tensor(self.rewards[indices], device=self.device),
            'next_remote_states': torch.tensor(self.next_remote_states[indices], device=self.device),
            'dones': torch.tensor(self.dones[indices], device=self.device),
            'delayed_sequences': torch.tensor(self.delayed_sequences[indices], device=self.device),
            'true_targets': torch.tensor(self.true_targets[indices], device=self.device),
            'next_delayed_sequences': torch.tensor(self.next_delayed_sequences[indices], device=self.device)
        }
    
    def __len__(self) -> int:
        return self.size
    
    
class SACTrainer:
    def __init__(self, 
                 env: VecEnv,
                 val_delay_config: ExperimentConfig = ExperimentConfig.LOW_DELAY
                 ):
        
        self.env = env
        self.num_envs = env.num_envs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Validation Env
        self.val_env = TeleoperationEnvWithDelay(
            delay_config=val_delay_config,
            trajectory_type=TrajectoryType.FIGURE_8,
            randomize_trajectory=False,
            render_mode=None,
            lstm_model_path=cfg.LSTM_MODEL_PATH 
        )
        
        # LSTM
        self.estimator = StateEstimator().to(self.device)
        # Load pre-trained weights to speed up convergence, but keep it trainable
        if os.path.exists(cfg.LSTM_MODEL_PATH):
            ckpt = torch.load(cfg.LSTM_MODEL_PATH, map_location=self.device, weights_only=False)
            if 'state_estimator_state_dict' in ckpt:
                self.estimator.load_state_dict(ckpt['state_estimator_state_dict'])
            else:
                self.estimator.load_state_dict(ckpt)
            print("[Trainer] Pre-trained LSTM weights loaded. Gradients ENABLED.")
        
        self.estimator_optimizer = torch.optim.Adam(self.estimator.parameters(), lr=1e-4)

        # SAC Agent
        self.actor = Actor(state_dim=cfg.OBS_DIM, action_dim=cfg.N_JOINTS).to(self.device)
        self.critic = Critic(state_dim=cfg.OBS_DIM, action_dim=cfg.N_JOINTS).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)

        for param in self.critic_target.parameters(): param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.SAC_LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.SAC_LEARNING_RATE)

        # Entropy
        if cfg.SAC_TARGET_ENTROPY == 'auto':
            self.target_entropy = -float(cfg.N_JOINTS)
        else:
            self.target_entropy = float(cfg.SAC_TARGET_ENTROPY)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.ALPHA_LEARNING_RATE)
        
        # 3. End-to-End Replay Buffer
        self.replay_buffer = ReplayBuffer(cfg.SAC_BUFFER_SIZE, self.device)

        # Constants for Patching
        self.dt_norm = (1.0 / cfg.DEFAULT_CONTROL_FREQ) / cfg.DELAY_INPUT_NORM_FACTOR
        
        # Training State
        self.total_timesteps = 0
        self.num_updates = 0
        self.checkpoint_dir = cfg.CHECKPOINT_DIR_RL
        self.tb_writer: Optional[SummaryWriter] = None
        
        # Logging
        self.best_validation_reward = -np.inf
        self.patience_counter = 0

    def _init_tensorboard(self):
        tb_dir = os.path.join(self.checkpoint_dir, "tensorboard_e2e")
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=120)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().detach().item()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_t, _, _ = self.actor.sample(obs_t, deterministic)
            return action_t.cpu().numpy()

    def _batch_autoregressive_inference(self, batch_sequences: torch.Tensor) -> torch.Tensor:
        """
        Batched AR Inference on GPU.
        Input: (Batch, SeqLen, 15)
        Output: (Batch, 14) -> Fresh Predicted q and qd
        """
        batch_size = batch_sequences.shape[0]
        
        with torch.no_grad():
            _, hidden_state = self.estimator.lstm(batch_sequences)

        # 2. Prepare Anchor
        last_obs = batch_sequences[:, -1, :]
        curr_q = last_obs[:, :cfg.N_JOINTS].clone()
        curr_qd = last_obs[:, cfg.N_JOINTS:2*cfg.N_JOINTS].clone()
        
        # Delay scalar (Batch, 1)
        # Assuming last element is normalized delay
        curr_delay = last_obs[:, -1].unsqueeze(1).clone() 

        # Determine steps to run (Simplify: run fixed max steps or estimate based on delay)
        # For batch efficiency, we assume worst-case or fixed steps.
        # We can calculate max delay in the batch.
        max_delay_norm = curr_delay.max().item()
        steps_to_run = int((max_delay_norm * cfg.DELAY_INPUT_NORM_FACTOR) * cfg.DEFAULT_CONTROL_FREQ)
        steps_to_run = min(steps_to_run + 2, cfg.MAX_AR_STEPS) # Safety buffer

        # 3. AR Loop
        for _ in range(steps_to_run):
            # Input: (Batch, 1, 15)
            ar_input = torch.cat([curr_q, curr_qd, curr_delay], dim=1).unsqueeze(1)
            
            # Predict
            residual_t, hidden_state = self.estimator.forward_step(ar_input, hidden_state)
            
            # Scale & Clamp
            residual = residual_t * cfg.TARGET_DELTA_SCALE # (Batch, 14)
            residual = torch.clamp(residual, -0.2, 0.2)
            
            # Update State
            curr_q = curr_q + residual[:, :cfg.N_JOINTS]
            curr_qd = curr_qd + residual[:, cfg.N_JOINTS:]
            
            # Update Delay
            curr_delay = curr_delay + self.dt_norm

        return torch.cat([curr_q, curr_qd], dim=1)

    def _patch_observations(self, obs_batch: torch.Tensor, fresh_preds: torch.Tensor) -> torch.Tensor:
        """
        Overwrites the 'Predicted' and 'Error' parts of the observation with fresh data.
        
        Obs Structure (113D):
        0-14: Remote (7q, 7qd)
        14-84: History (70)
        84-98: Predicted (14)  <-- PATCH THIS
        98-112: Error (14)     <-- PATCH THIS
        112: Delay (1)
        """
        patched_obs = obs_batch.clone()
        
        # Extract Remote State (for error calc)
        remote_state = patched_obs[:, :14]
        
        # Calculate Fresh Error
        fresh_error = fresh_preds - remote_state
        
        # Overwrite Prediction (Indices 84 to 98)
        patched_obs[:, 84:98] = fresh_preds
        
        # Overwrite Error (Indices 98 to 112)
        patched_obs[:, 98:112] = fresh_error
        
        return patched_obs

    def update_end_to_end(self) -> Dict[str, float]:
        if len(self.replay_buffer) < cfg.SAC_BATCH_SIZE:
            return {}
        
        batch = self.replay_buffer.sample(cfg.SAC_BATCH_SIZE)
        
        # =====================================================================
        # STEP 1: Update Estimator (Supervised Learning)
        # =====================================================================
        # We predict 1 step into the future or perform AR training. 
        # For simplicity and stability, we train on the LSTM's ability to predict 
        # the *immediate* next residuals or the final target in the sequence.
        # Here we assume 'true_targets' is the ground truth for the END of the AR chain.
        
        # Note: Training AR on a batch with varying delays is complex.
        # Strategy: Use the same AR rollout logic but with Gradients ENABLED.
        
        # Run AR with gradients
        pred_targets_grad = self._batch_autoregressive_inference(batch['delayed_sequences'])
        
        estimator_loss = F.l1_loss(pred_targets_grad, batch['true_targets'])
        
        self.estimator_optimizer.zero_grad()
        estimator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), 1.0)
        self.estimator_optimizer.step()
        
        # =====================================================================
        # STEP 2: Data Patching (Fresh Predictions for SAC)
        # =====================================================================
        with torch.no_grad():
            # Get fresh predictions for Current State (using updated LSTM)
            fresh_preds_curr = self._batch_autoregressive_inference(batch['delayed_sequences'])
            patched_obs = self._patch_observations(batch['remote_states'], fresh_preds_curr)
            
            # Get fresh predictions for Next State
            fresh_preds_next = self._batch_autoregressive_inference(batch['next_delayed_sequences'])
            patched_next_obs = self._patch_observations(batch['next_remote_states'], fresh_preds_next)

        # IMPORTANT: Detach patched obs so SAC gradients don't flow into LSTM
        # (Although no_grad() handles this, explicit detach is safer for logic clarity)
        sac_obs = patched_obs.detach()
        sac_next_obs = patched_next_obs.detach()

        # =====================================================================
        # STEP 3: SAC Update (RL Learning)
        # =====================================================================
        
        # Critic Target
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(sac_next_obs)
            q1_target, q2_target = self.critic_target(sac_next_obs, next_actions)
            q_min = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            q_target = batch['rewards'] + (1 - batch['dones']) * cfg.SAC_GAMMA * q_min

        # Critic Update
        q1, q2 = self.critic(sac_obs, batch['actions'])
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        actions, log_probs, _ = self.actor.sample(sac_obs)
        q1_pi, q2_pi = self.critic(sac_obs, actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_probs - q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha Update
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Target Soft Update
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1 - cfg.SAC_TAU).add_(cfg.SAC_TAU * p.data)

        return {
            'estimator_loss': estimator_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha
        }

    def train(self, total_timesteps: int):
        self._init_tensorboard()
        logger.info("Starting End-to-End SAC Training...")
        
        obs = self.env.reset()
        episode_rewards = np.zeros(self.num_envs)
        
        # Need to fetch initial delayed sequences from Env
        # We assume env has method `get_delayed_target_buffer` that returns the flattened seq
        # We need to reshape it: (Batch, SeqLen, 15)
        raw_seqs_flat = self.env.env_method("get_delayed_target_buffer", cfg.RNN_SEQUENCE_LENGTH)
        current_seqs = np.array(raw_seqs_flat).reshape(self.num_envs, cfg.RNN_SEQUENCE_LENGTH, -1)
        
        for t in range(int(total_timesteps // self.num_envs)):
            
            # Action Selection
            if self.total_timesteps < cfg.SAC_START_STEPS:
                actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
            else:
                actions = self.select_action(obs)

            # Step
            next_obs, rewards, dones, infos = self.env.step(actions)
            
            # Get Next Sequences and True Targets for Buffer
            next_raw_seqs_flat = self.env.env_method("get_delayed_target_buffer", cfg.RNN_SEQUENCE_LENGTH)
            next_seqs = np.array(next_raw_seqs_flat).reshape(self.num_envs, cfg.RNN_SEQUENCE_LENGTH, -1)
            
            # Get Ground Truth Targets (for Estimator Supervision)
            true_targets = np.array(self.env.env_method("get_true_current_target"))
            # Truncate to 14D (q+qd) just in case
            true_targets = true_targets[:, :14]

            for i in range(self.num_envs):
                if infos[i].get('is_in_warmup', False): continue
                
                # ADD TO BUFFER: RL Data + LSTM Data
                self.replay_buffer.add(
                    obs[i], actions[i], rewards[i], next_obs[i], dones[i],
                    current_seqs[i], true_targets[i], next_seqs[i]
                )
            
            obs = next_obs
            current_seqs = next_seqs
            episode_rewards += rewards
            self.total_timesteps += self.num_envs
            
            # Update
            if self.total_timesteps >= cfg.SAC_START_STEPS:
                for _ in range(int(cfg.SAC_UPDATES_PER_STEP * self.num_envs)):
                    metrics = self.update_end_to_end()
                    self.num_updates += 1

            # Logging & Validation (Same as before)
            if (t+1) % 500 == 0:
                 avg_r = np.mean(episode_rewards)
                 if self.tb_writer:
                     self.tb_writer.add_scalar('train/reward', avg_r, self.total_timesteps)
                     if 'estimator_loss' in locals().get('metrics', {}):
                         self.tb_writer.add_scalar('train/est_loss', metrics['estimator_loss'], self.total_timesteps)
                 episode_rewards = np.zeros(self.num_envs)

            if (t + 1) % (cfg.SAC_VAL_FREQ // self.num_envs) == 0:
                self.save_checkpoint("latest_e2e_model.pth")
                
        self.save_checkpoint("final_e2e_model.pth")

    def save_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'estimator_state_dict': self.estimator.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)    