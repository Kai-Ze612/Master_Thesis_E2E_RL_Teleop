"""
Pre-training script for the Autoregressive State Estimator (Closed-Loop LSTM).
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from datetime import datetime
import torch.nn.functional as F
from typing import Dict, Tuple
import multiprocessing

from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter


from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    RNN_SEQUENCE_LENGTH,
    ESTIMATOR_LEARNING_RATE,
    ESTIMATOR_BATCH_SIZE,
    ESTIMATOR_BUFFER_SIZE,
    ESTIMATOR_TOTAL_UPDATES,
    NUM_ENVIRONMENTS,
    CHECKPOINT_DIR_LSTM,
    TARGET_DELTA_SCALE,
    DT,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    DELAY_INPUT_NORM_FACTOR,
    ESTIMATOR_VAL_FREQ,
    ESTIMATOR_PATIENCE,
    ESTIMATOR_STATE_DIM,
)


class AutoregressiveStateEstimator(nn.Module):
    def __init__(self, input_dim_total=15, output_dim=14):
        """
        input_dim: 14D(q + qd) + 1D (normalized delay)
        output_dim: 14D (predicted q + qd)
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim_total,
            hidden_size=RNN_HIDDEN_DIM, 
            num_layers=RNN_NUM_LAYERS,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(RNN_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        residual = self.fc(last_hidden)
        return residual, None

    def forward_step(self, x_step, hidden_state):
        lstm_out, new_hidden = self.lstm(x_step, hidden_state)
        residual = self.fc(lstm_out[:, -1, :])
        return residual, new_hidden

def is_trajectory_stable(delayed_seq: np.ndarray, true_target: np.ndarray) -> bool:
    # Check for NaNs
    if np.isnan(delayed_seq).any() or np.isnan(true_target).any():
        return False
    
    # Check for large jumps in joint positions/velocities
    if np.max(np.abs(delayed_seq[:, 7:14])) > 6.0:
        return False 
    
    return True

def setup_logging(output_dir: str) -> logging.Logger:
    log_file = os.path.join(output_dir, "autoregressive_train.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0  # Pointer to the next insertion index
        self.size = 0
        self.seq_len = RNN_SEQUENCE_LENGTH
        self.state_dim = ESTIMATOR_STATE_DIM
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.state_dim), dtype=np.float32)
        self.true_targets = np.zeros((buffer_size, 14), dtype=np.float32)

    def add(self, delayed_seq: np.ndarray, true_target: np.ndarray) -> None:
        self.delayed_sequences[self.ptr] = delayed_seq
        self.true_targets[self.ptr] = true_target
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'delayed_sequences': torch.tensor(self.delayed_sequences[indices], device=self.device),
            'true_targets': torch.tensor(self.true_targets[indices], device=self.device),
        }
    def __len__(self) -> int: return self.size

def collect_data_from_envs(env: SubprocVecEnv, num_envs: int) -> Tuple[np.ndarray, np.ndarray]:
    # Get data from Env
    delayed_flat_list = env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
    true_target_list = env.env_method("get_true_current_target")
    
    # Reshape
    raw_seqs = np.array([buf.reshape(RNN_SEQUENCE_LENGTH, 15) for buf in delayed_flat_list])
    raw_targets = np.array(true_target_list)
    
    valid_seqs, valid_targets = [], []
    for i in range(num_envs):
        if is_trajectory_stable(raw_seqs[i], raw_targets[i]):
            valid_seqs.append(raw_seqs[i])
            valid_targets.append(raw_targets[i])
            
    if len(valid_seqs) == 0: return np.array([]), np.array([])
    return np.array(valid_seqs), np.array(valid_targets)

def evaluate_model(model, val_env, num_val_steps=50, device='cpu'):
    """Run validation loop without training to check generalization."""
    model.eval()
    total_loss = 0.0
    count = 0
    
    # Reset Validation Envs
    val_env.reset()
    # Warmup
    for _ in range(20): val_env.step([np.zeros((val_env.num_envs, N_JOINTS))])
    
    with torch.no_grad():
        for _ in range(num_val_steps):
            seqs, targets = collect_data_from_envs(val_env, val_env.num_envs)
            if len(seqs) == 0: 
                val_env.step([np.zeros((val_env.num_envs, N_JOINTS))])
                continue
            
            input_t = torch.tensor(seqs, dtype=torch.float32).to(device)
            target_t = torch.tensor(targets, dtype=torch.float32).to(device)
            
            # Standard Forward (Open Loop) for validation metric
            # Or should we validate Closed Loop? Usually standard forward is a good proxy for stability.
            pred_residual, _ = model(input_t)
            
            # Reconstruct absolute state
            # The model predicts residual relative to the END of the input sequence
            last_obs = input_t[:, -1, :14]
            pred_state = last_obs + (pred_residual / TARGET_DELTA_SCALE) # Descale
            
            loss = F.mse_loss(pred_state, target_t)
            total_loss += loss.item()
            count += 1
            
            val_env.step([np.zeros((val_env.num_envs, N_JOINTS))])
            
    model.train()
    return total_loss / max(1, count)

# ----------------------------------------------------------------------------
# 3. Main Training
# ----------------------------------------------------------------------------
def train_autoregressive_estimator(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Autoregressive_LSTM_{args.config.name}_{timestamp}"
    output_dir = os.path.join(CHECKPOINT_DIR_LSTM, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)    
    logger.info("="*80)
    logger.info("AUTOREGRESSIVE LSTM TRAINING (With Early Stopping)")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    # 1. Training Envs
    def make_env(rank: int):
        def _init():
            return TeleoperationEnvWithDelay(
                delay_config=args.config,
                trajectory_type=args.trajectory_type,
                randomize_trajectory=args.randomize_trajectory,
                seed=args.seed + rank,
                lstm_model_path=None # Passthrough mode
            )
        return _init
    
    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVIRONMENTS)])
    
    # 2. Validation Envs (Different Seeds)
    val_env = SubprocVecEnv([make_env(i + 1000) for i in range(NUM_ENVIRONMENTS)])
    
    replay_buffer = ReplayBuffer(ESTIMATOR_BUFFER_SIZE, device)
    model = AutoregressiveStateEstimator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ESTIMATOR_LEARNING_RATE)
    
    # Initial Collection
    logger.info("Collecting Initial Data...")
    train_env.reset()
    for _ in range(100): train_env.step([np.zeros((NUM_ENVIRONMENTS, N_JOINTS))]) 
    
    collected = 0
    while collected < ESTIMATOR_BUFFER_SIZE // 2:
        seqs, targets = collect_data_from_envs(train_env, NUM_ENVIRONMENTS)
        if len(seqs) > 0:
            for i in range(len(seqs)): replay_buffer.add(seqs[i], targets[i])
            collected += len(seqs)
            train_env.step([np.zeros((NUM_ENVIRONMENTS, N_JOINTS))])
        else:
            train_env.reset()
            for _ in range(50): train_env.step([np.zeros((NUM_ENVIRONMENTS, N_JOINTS))])

    logger.info(f"Buffer filled. Starting Training Loop.")
    
    # --- Early Stopping Vars ---
    best_val_loss = float('inf')
    patience_counter = 0
    
    model.train()
    
    for update in range(1, ESTIMATOR_TOTAL_UPDATES + 1):
        
        # A. Data Collection
        seqs, targets = collect_data_from_envs(train_env, NUM_ENVIRONMENTS)
        if len(seqs) > 0:
            for i in range(len(seqs)): replay_buffer.add(seqs[i], targets[i])
        train_env.step([np.zeros((NUM_ENVIRONMENTS, N_JOINTS))])
        
        # B. Training Step
        batch = replay_buffer.sample(ESTIMATOR_BATCH_SIZE)
        input_seq = batch['delayed_sequences']
        true_final_target = batch['true_targets']
        
        # Recursive Logic
        packet_loss_steps = np.random.randint(1, 15)
        cutoff_idx = RNN_SEQUENCE_LENGTH - packet_loss_steps
        safe_history = input_seq[:, :cutoff_idx, :]
        
        _, hidden_state = model.lstm(safe_history)
        current_input = safe_history[:, -1:, :]
        
        for k in range(packet_loss_steps):
            pred_delta, hidden_state = model.forward_step(current_input, hidden_state)
            
            last_known_state = current_input[:, :, :14] 
            predicted_next_state = last_known_state + (pred_delta.unsqueeze(1) * TARGET_DELTA_SCALE)
            
            current_delay_norm = current_input[:, :, 14]
            next_delay_norm = current_delay_norm + (DT / DELAY_INPUT_NORM_FACTOR)
            next_input = torch.cat([predicted_next_state, next_delay_norm.unsqueeze(2)], dim=2)
            current_input = next_input 
            
        final_predicted_state = current_input[:, 0, :14]
        loss = F.l1_loss(final_predicted_state, true_final_target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # C. Logging
        if update % 100 == 0:
            tb_writer.add_scalar('loss/train', loss.item(), update)
            print(f"Update {update} | Loss: {loss.item():.6f}")
            
        # D. Validation & Early Stopping
        if update % ESTIMATOR_VAL_FREQ == 0:
            val_loss = evaluate_model(model, val_env, device=device)
            tb_writer.add_scalar('loss/validation', val_loss, update)
            
            logger.info(f"Validating at {update}: Val Loss = {val_loss:.6f} (Best: {best_val_loss:.6f})")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save Best Model
                save_path = os.path.join(output_dir, "autoregressive_estimator.pth")
                torch.save({'state_estimator_state_dict': model.state_dict()}, save_path)
                logger.info(f"  -> New Best! Saved model.")
            else:
                patience_counter += 1
                logger.info(f"  -> No improvement. Patience: {patience_counter}/{ESTIMATOR_PATIENCE}")
                
                if patience_counter >= ESTIMATOR_PATIENCE:
                    logger.info("Early Stopping Triggered.")
                    break
            
    logger.info("Training Complete.")
    train_env.close()
    val_env.close()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="3", choices=['1','2','3','4'])
    parser.add_argument("--trajectory-type", type=str, default="figure_8", choices=[t.value for t in TrajectoryType])
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--randomize-trajectory", action="store_true")
    args = parser.parse_args()
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2], '4': config_options[3]}
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    return args

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    train_autoregressive_estimator(parse_arguments())