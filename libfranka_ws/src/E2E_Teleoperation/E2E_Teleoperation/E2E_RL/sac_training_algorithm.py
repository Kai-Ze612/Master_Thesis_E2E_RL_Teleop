"""
SAC training algorithm

Two stages training:
1. LSTM encoder pre-training (state estimator)
2. Policy distillation (teacher - student learning)
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, Optional

import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.sac_policy_network import JointActor, JointCritic, SharedLSTMEncoder

logger = logging.getLogger(__name__)


class EarlyStopper:
    """
    Handles early stopping based on validation reward/loss.
    """
    def __init__(self, patience: int = cfg.EARLY_STOP_PATIENCE):
        self.patience = patience
        self.counter = 0
        self.best_metric = -np.inf  # (Reward higher is better)
        
    def check(self, metric: float) -> Tuple[bool, bool]:
        if metric > self.best_metric:
            self.best_metric = metric
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
            return False, False
        
        
class ReplayBuffer:
    """
    Simple Replay Buffer for storing Expert (Teacher) Trajectories.
    """
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.teacher_actions = np.zeros((capacity, action_dim), dtype=np.float32) # Target Labels
        self.true_states = np.zeros((capacity, 14), dtype=np.float32) # For Encoder Loss
        
    def add(self, obs: np.ndarray, teacher_action: np.ndarray, true_state: np.ndarray):
        self.obs[self.ptr] = obs
        self.teacher_actions[self.ptr] = teacher_action
        self.true_states[self.ptr] = true_state
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idxs]).to(self.device),
            torch.FloatTensor(self.teacher_actions[idxs]).to(self.device),
            torch.FloatTensor(self.true_states[idxs]).to(self.device)
        )
    
    @property
    def current_size(self):
        return self.size
    

class PhasedTrainer:
    """
    Manages the two-stage training process:
    1. Encoder Training
    2. Student Distillation
    """
    def __init__(self, env, output_dir: str):
        self.env = env
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))
        
        # Networks
        self.shared_encoder = SharedLSTMEncoder().to(self.device)
        self.actor = JointActor(shared_encoder=self.shared_encoder, action_dim=cfg.N_JOINTS).to(self.device)
        self.critic = JointCritic(shared_encoder=self.shared_encoder, action_dim=cfg.N_JOINTS).to(self.device)
        
        # Optimizers
        self.enc_optimizer = optim.Adam(self.shared_encoder.parameters(), lr=cfg.ENCODER_LR)
        self.actor_optimizer = optim.Adam(
            [p for n, p in self.actor.named_parameters() if 'encoder' not in n], 
            lr=cfg.ACTOR_LR
        )
        
        # Data & Utils
        self.buffer = ReplayBuffer(cfg.BUFFER_SIZE, cfg.OBS_DIM, cfg.N_JOINTS, self.device)
        self.stopper = EarlyStopper()
        self.global_step = 0

    def collect_expert_data(self, steps: int, use_prediction: bool = False):
        
        obs, _ = self.env.reset()
        
        for _ in range(steps):
            # 1. Feed prediction back to env for State Estimation robustness
            if use_prediction:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    # [FIX 1] Unpack 4 values (action, log_prob, raw, pred)
                    _, _, _, pred = self.actor.sample(obs_t, deterministic=True)
                    pred_np = pred.cpu().numpy()[0]
                self.env.set_predicted_state(pred_np)
 
            # Extract raw state data needed for Teacher Calculation
            r_q, r_qd = self.env.remote.get_joint_state()
            target_q, target_qd = self.env.leader_hist[-1] # Ground Truth Target
            
            # Compute Optimal Torque (Label)
            teacher_action = self.env._compute_total_torque(r_q, r_qd, target_q, target_qd)
            
            # Apply Teacher Action (Drive the robot perfectly)
            next_obs, reward, done, _, info = self.env.step(teacher_action)
            true_state = info['true_state']
            
            # Store Transition
            self.buffer.add(obs, teacher_action, true_state)
            
            obs = next_obs
            if done: 
                obs, _ = self.env.reset()

    def train_stage_1_encoder(self):
        """
        Stage 1: Train LSTM to predict current state from delayed history.
        Policy is frozen.
        """
        logger.info("="*60)
        logger.info(">>> STARTING STAGE 1: ENCODER PRE-TRAINING")
        logger.info("="*60)
        
        # Freeze Policy Backbone
        for param in self.actor.backbone.parameters(): param.requires_grad = False
        self.actor.fc_mean.requires_grad = False
        self.actor.fc_log_std.requires_grad = False
        
        for step in range(cfg.STAGE1_STEPS):
            # Collect data periodically
            if self.buffer.current_size < cfg.BATCH_SIZE or step % 5000 == 0:
                self.collect_expert_data(steps=1000, use_prediction=False)
            
            # Sample
            obs, _, true_states = self.buffer.sample(cfg.BATCH_SIZE)
            
            # Prepare Inputs
            target_history = obs[:, -cfg.TARGET_HISTORY_DIM:]
            history_seq = target_history.view(-1, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
            
            # Forward Encoder
            # [FIX 2] Unpack 3 values (features, pred_state, hidden)
            _, pred_state, _ = self.shared_encoder(history_seq)
            
            # Prepare Targets
            t_q = true_states[:, :7]
            t_qd = true_states[:, 7:]
            t_q_norm = (t_q - torch.tensor(cfg.Q_MEAN, device=self.device)) / torch.tensor(cfg.Q_STD, device=self.device)
            t_qd_norm = (t_qd - torch.tensor(cfg.QD_MEAN, device=self.device)) / torch.tensor(cfg.QD_STD, device=self.device)
            target = torch.cat([t_q_norm, t_qd_norm], dim=1)
            
            # Loss & Update
            loss = F.mse_loss(pred_state, target)
            
            self.enc_optimizer.zero_grad()
            loss.backward()
            self.enc_optimizer.step()
            
            if step % 1000 == 0:
                logger.info(f"[Stage 1] Step {step} | Encoder Loss: {loss.item():.6f}")
                self.writer.add_scalar("Stage1/Loss", loss.item(), step)

        # Unfreeze Policy for Stage 2
        for param in self.actor.parameters(): param.requires_grad = True
        logger.info(">>> STAGE 1 COMPLETE")

    def train_stage_2_distillation(self):
        """
        Stage 2: Train Student (Actor) to mimic Teacher (ID Controller).
        Uses Behavioral Cloning (MSE Loss).
        """
        logger.info("="*60)
        logger.info(">>> STARTING STAGE 2: TEACHER DISTILLATION (BC)")
        logger.info("="*60)
        
        self.global_step = 0
        
        while self.global_step < cfg.STAGE2_TOTAL_STEPS:
            # 1. Collect Data (Teacher drives, but we can use Student prediction for state estimation)
            self.collect_expert_data(steps=200, use_prediction=True)
            self.global_step += 200
            
            # 2. Train Loop
            for _ in range(50): 
                obs, teacher_actions, _ = self.buffer.sample(cfg.BATCH_SIZE)
                
                # Forward Student (Deterministic)
                # Unpack 4 values: (action, log_prob, raw, pred)
                student_action, _, _, _ = self.actor.sample(obs, deterministic=True)
                
                # Behavioral Cloning Loss (MSE)
                bc_loss = F.mse_loss(student_action, teacher_actions)
                
                # Update Student & Fine-tune Encoder
                self.actor_optimizer.zero_grad()
                self.enc_optimizer.zero_grad()
                
                bc_loss.backward()
                
                self.actor_optimizer.step()
                self.enc_optimizer.step()
            
            # 3. Validation & Logging
            if self.global_step % cfg.LOG_FREQ == 0:
                logger.info(f"[Stage 2] Step {self.global_step} | BC Loss: {bc_loss.item():.6f}")
                self.writer.add_scalar("Stage2/BC_Loss", bc_loss.item(), self.global_step)
            
            if self.global_step % cfg.VAL_FREQ == 0:
                val_reward = self.validate()
                stop, is_best = self.stopper.check(val_reward)
                
                logger.info(f"[Validation] Step {self.global_step} | Mean Reward: {val_reward:.2f}")
                self.writer.add_scalar("Validation/Reward", val_reward, self.global_step)
                
                if is_best:
                    self.save_checkpoint("best_policy.pth")
                
                if stop:
                    logger.info(">>> Early Stopping Triggered. Training Finished.")
                    break

    def validate(self) -> float:
        """
        Evaluates the Student Policy in the environment.
        The Student drives the robot here.
        """
        total_reward = 0.0
        
        for _ in range(cfg.VAL_EPISODES):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    # Unpack 4 values: (action, log_prob, raw, pred)
                    action, _, _, pred = self.actor.sample(obs_t, deterministic=True)
                    
                    action_np = action.cpu().numpy()[0]
                    pred_np = pred.cpu().numpy()[0]
                
                self.env.set_predicted_state(pred_np)
                
                # Step Environment with STUDENT action
                next_obs, reward, done, _, _ = self.env.step(action_np)
                
                ep_reward += reward
                obs = next_obs
            
            total_reward += ep_reward
            
        return total_reward / cfg.VAL_EPISODES

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'actor': self.actor.state_dict(),
            'encoder': self.shared_encoder.state_dict(),
            'optimizer': self.actor_optimizer.state_dict()
        }, path)