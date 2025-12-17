"""
SAC training algorithm

Two stages training:
1. LSTM encoder pre-training (state estimator)
2. ID Learning
3. Policy Distallation (Teacher - Student learning)
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
    def __init__(self, patience: int = cfg.EARLY_STOP_PATIENCE):
        self.patience = patience
        self.counter = 0
        self.best_metric = -np.inf 
        
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
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.teacher_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.true_states = np.zeros((capacity, 14), dtype=np.float32)
        
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
    def __init__(self, env, output_dir: str):
        self.env = env
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))
        
        self.shared_encoder = SharedLSTMEncoder().to(self.device)
        self.actor = JointActor(shared_encoder=self.shared_encoder, action_dim=cfg.N_JOINTS).to(self.device)
        self.critic = JointCritic(shared_encoder=self.shared_encoder, action_dim=cfg.N_JOINTS).to(self.device)
        
        self.enc_optimizer = optim.Adam(self.shared_encoder.parameters(), lr=cfg.ENCODER_LR)
        self.actor_optimizer = optim.Adam(
            [p for n, p in self.actor.named_parameters() if 'encoder' not in n], 
            lr=cfg.ACTOR_LR
        )
        
        self.buffer = ReplayBuffer(cfg.BUFFER_SIZE, cfg.OBS_DIM, cfg.N_JOINTS, self.device)
        self.stopper = EarlyStopper()
        self.global_step = 0

    def collect_expert_data(self, steps: int, use_student_driver: bool = False):
        """
        Collects data for the replay buffer. 
        Supports DAgger (Student drives) and Teacher Forcing.
        Includes debug prints to verify Teacher strength.
        """
        obs, _ = self.env.reset()
        
        for i in range(steps):
            # 1. Decide who drives the PHYSICAL robot
            if use_student_driver:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    # Student drives (deterministic=True is standard for DAgger)
                    action, _, _, pred = self.actor.sample(obs_t, deterministic=True)
                    
                    # Denormalize prediction for Env visualization/logging
                    pred_np = pred.cpu().numpy()[0]
                    pred_denorm = np.zeros_like(pred_np)
                    pred_denorm[:7] = pred_np[:7] * cfg.Q_STD + cfg.Q_MEAN
                    pred_denorm[7:] = pred_np[7:] * cfg.QD_STD + cfg.QD_MEAN
                    self.env.set_predicted_state(pred_denorm)
                    
                    # The physical action executed in the env
                    driving_action = action.cpu().numpy()[0]
            else:
                # Teacher drives (Perfect Trajectory)
                r_q, r_qd = self.env.remote.get_joint_state()
                t_q, t_qd = self.env.leader_hist[-1]
                driving_action = self.env._compute_teacher_torque(r_q, r_qd, t_q, t_qd)

            # 2. ALWAYS Calculate Teacher Label (The "Correct" Action)
            # This is what we save to the buffer as the "Target" for the loss function
            r_q_curr, r_qd_curr = self.env.remote.get_joint_state()
            target_q, target_qd = self.env.leader_hist[-1]
            teacher_label = self.env._compute_teacher_torque(r_q_curr, r_qd_curr, target_q, target_qd)
            
            # DEBUG PRINT
            if i % 100 == 0:
                print(f"Step {i} Teacher Torque: {np.round(teacher_label, 3)}")
            
            # Only print for the first 5 steps of the collection phase to avoid spam
            if i < 5: 
                # Pick a specific joint to monitor (e.g., Joint 4 which fights gravity)
                j_idx = 3 
                print(f"[Teacher Debug] Step {i}")
                print(f"  Robot Q[{j_idx}]:     {r_q_curr[j_idx]:.3f}")
                print(f"  Teacher Tau[{j_idx}]: {teacher_label[j_idx]:.3f} (Should be > 10.0 for gravity)")
                print(f"  Driving Tau[{j_idx}]: {driving_action[j_idx]:.3f}")
                print("-" * 40)

            # 3. Step Env
            next_obs, reward, done, _, info = self.env.step(driving_action)
            true_state = info['true_state']
            
            # 4. Save to Buffer: (Obs, Teacher_Label, True_State)
            # Crucial: We save 'teacher_label', NOT 'driving_action' as the target!
            self.buffer.add(obs, teacher_label, true_state)
            
            obs = next_obs
            if done: 
                obs, _ = self.env.reset()

    def train_stage_1_encoder(self):
        """Stage 1: Encoder Pre-training."""
        logger.info("="*60)
        logger.info(">>> STAGE 1: ENCODER PRE-TRAINING")
        logger.info("="*60)
        
        # Freeze Policy
        for param in self.actor.backbone.parameters(): param.requires_grad = False
        self.actor.fc_mean.requires_grad = False
        self.actor.fc_log_std.requires_grad = False
        
        updates_completed = 0
        COLLECTION_STEPS = 5000 
        
        while updates_completed < cfg.STAGE1_STEPS:
            # 1. Collect Data (Teacher always drives in Stage 1)
            logger.info(f"Collecting {COLLECTION_STEPS} steps of expert data...")
            
            # [FIXED] Updated argument name
            self.collect_expert_data(steps=COLLECTION_STEPS, use_student_driver=False)
            
            # 2. Train
            logger.info(f"Training for {COLLECTION_STEPS} updates...")
            train_steps = COLLECTION_STEPS
            
            for _ in range(train_steps):
                if updates_completed >= cfg.STAGE1_STEPS: break
                
                obs, _, true_states = self.buffer.sample(cfg.BATCH_SIZE)
                target_history = obs[:, -cfg.TARGET_HISTORY_DIM:]
                history_seq = target_history.view(-1, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
                
                _, pred_state, _ = self.shared_encoder(history_seq)
                
                t_q = true_states[:, :7]
                t_qd = true_states[:, 7:]
                t_q_norm = (t_q - torch.tensor(cfg.Q_MEAN, device=self.device)) / torch.tensor(cfg.Q_STD, device=self.device)
                t_qd_norm = (t_qd - torch.tensor(cfg.QD_MEAN, device=self.device)) / torch.tensor(cfg.QD_STD, device=self.device)
                target = torch.cat([t_q_norm, t_qd_norm], dim=1)
                
                loss = F.mse_loss(pred_state, target)
                
                self.enc_optimizer.zero_grad()
                loss.backward()
                self.enc_optimizer.step()
                
                updates_completed += 1
                if updates_completed % 1000 == 0:
                    logger.info(f"[Stage 1] Update {updates_completed} | Loss: {loss.item():.6f}")
                    self.writer.add_scalar("Stage1/Loss", loss.item(), updates_completed)

        # Unfreeze Policy
        for param in self.actor.parameters(): param.requires_grad = True
        logger.info(">>> STAGE 1 COMPLETE")

    def train_stage_2_distillation(self):
        """Stage 2: Teacher Distillation (DAgger)."""
        logger.info("="*60)
        logger.info(">>> STAGE 2: TEACHER DISTILLATION (DAgger)")
        logger.info("="*60)
        
        self.global_step = 0
        COLLECTION_STEPS = 5000
        
        while self.global_step < cfg.STAGE2_TOTAL_STEPS:
            # DAgger Strategy:
            # Initially, let Teacher drive (warm start).
            # As training progresses, let Student drive more often to learn recovery.
            use_student = (self.global_step > 20000)
            
            driver_name = "Student (DAgger)" if use_student else "Teacher (Expert)"
            logger.info(f"Collecting {COLLECTION_STEPS} steps | Driver: {driver_name}")
            
            # [FIXED] Using correct argument
            self.collect_expert_data(steps=COLLECTION_STEPS, use_student_driver=use_student)
            
            self.global_step += COLLECTION_STEPS
            
            logger.info(f"Training for {COLLECTION_STEPS} updates...")
            
            for i in range(COLLECTION_STEPS): 
                obs, teacher_actions, _ = self.buffer.sample(cfg.BATCH_SIZE)
                
                # Student outputs SCALED action (Nm)
                student_action, _, _, _ = self.actor.sample(obs, deterministic=True)
                
                # [FIX] Normalization Fix:
                # We normalize both student and teacher actions to [-1, 1] range for the loss.
                # Calculating loss on raw Torque (e.g. 87.0) is numerically unstable.
                scale = self.actor.action_scale
                
                # Note: student_action / scale essentially recovers the 'tanh' output
                # teacher_actions are loaded from buffer in raw Nm
                bc_loss = F.mse_loss(student_action / scale, teacher_actions / scale)
                
                self.actor_optimizer.zero_grad()
                self.enc_optimizer.zero_grad()
                bc_loss.backward()
                self.actor_optimizer.step()
                self.enc_optimizer.step()
            
                if i % 1000 == 0:
                    logger.info(f"[Stage 2] Batch Loss: {bc_loss.item():.6f}")

            # 3. Validation
            if self.global_step % cfg.VAL_FREQ == 0:
                val_reward = self.validate()
                stop, is_best = self.stopper.check(val_reward)
                
                logger.info(f"[Validation] Step {self.global_step} | Mean Reward: {val_reward:.2f}")
                self.writer.add_scalar("Validation/Reward", val_reward, self.global_step)
                
                if is_best:
                    self.save_checkpoint("best_policy.pth")
                    logger.info(">>> New Best Model Saved!")
                
                if stop:
                    logger.info(">>> Early Stopping Triggered.")
                    break

    def validate(self) -> float:
        """
        Runs validation with Debug prints to verify LSTM and Torque saturation.
        """
        total_reward = 0.0
        self.actor.eval() # Set eval mode
        
        debug_limit = 5 
        
        for ep_idx in range(cfg.VAL_EPISODES):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            
            while not done:
                with torch.no_grad():
                    # 1. Inference
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _, _, pred = self.actor.sample(obs_t, deterministic=True)
                    
                    action_np = action.cpu().numpy()[0]
                    pred_np = pred.cpu().numpy()[0]

                # 2. Denormalize Prediction for Env/Logging
                pred_denorm = np.zeros_like(pred_np)
                pred_denorm[:7] = pred_np[:7] * cfg.Q_STD + cfg.Q_MEAN
                pred_denorm[7:] = pred_np[7:] * cfg.QD_STD + cfg.QD_MEAN
                
                self.env.set_predicted_state(pred_denorm)

                # 3. [DEBUG] Compare True vs Pred vs Action
                if ep_idx == 0 and step < debug_limit:
                    true_q, _ = self.env.leader_hist[-1]
                    
                    print(f"\n[Validation Debug] Step {step}")
                    print(f"  True Pose:   {np.round(true_q, 3)}")
                    print(f"  LSTM Pred:   {np.round(pred_denorm[:7], 3)}")
                    print(f"  Action (Nm): {np.round(action_np, 2)}")
                    
                    if np.any(np.abs(action_np) >= (cfg.MAX_ACTION_TORQUE - 0.1)):
                         print("  >>> WARNING: Action saturated! Robot may be too weak to lift arm.")

                # 4. Step
                next_obs, reward, done, _, info = self.env.step(action_np)
                ep_reward += reward
                obs = next_obs
                step += 1
            
            total_reward += ep_reward
            
        self.actor.train() # Reset to train mode
        return total_reward / cfg.VAL_EPISODES

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'actor': self.actor.state_dict(),
            'encoder': self.shared_encoder.state_dict(),
            'optimizer': self.actor_optimizer.state_dict()
        }, path)