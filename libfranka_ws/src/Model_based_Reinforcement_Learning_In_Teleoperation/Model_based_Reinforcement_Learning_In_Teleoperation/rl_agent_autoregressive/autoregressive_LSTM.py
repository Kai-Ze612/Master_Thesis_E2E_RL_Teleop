"""
Pre-training script for the Autoregressive State Estimator (Closed-Loop LSTM).

Key Features:
1. Scheduled Sampling: Randomly switches between "Real Data" and "Self-Prediction" during training.
2. Recursive Inference: Explicitly loops the LSTM hidden states to simulate packet loss.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from datetime import datetime
from collections import deque
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import multiprocessing
import random

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Adjust imports to your folder structure
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    RNN_SEQUENCE_LENGTH,
    ESTIMATOR_LEARNING_RATE,
    ESTIMATOR_BATCH_SIZE,
    ESTIMATOR_BUFFER_SIZE,
    ESTIMATOR_TOTAL_UPDATES,
    NUM_ENVIRONMENTS,
    CHECKPOINT_DIR_LSTM,
    ESTIMATOR_VAL_FREQ,
    ESTIMATOR_PATIENCE,
    ESTIMATOR_LR_PATIENCE,
    INITIAL_JOINT_CONFIG,
    TARGET_DELTA_SCALE,
    DEFAULT_CONTROL_FREQ,
    DEFAULT_PUBLISH_FREQ,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    DELAY_INPUT_NORM_FACTOR,
    DT, # Need DT to increment delay during recursion
)

# ----------------------------------------------------------------------------
# 1. Modified LSTM Model for Recursive Steps
# ----------------------------------------------------------------------------
class AutoregressiveStateEstimator(nn.Module):
    def __init__(self, input_dim_total=N_JOINTS*2 + 1, output_dim=N_JOINTS*2):
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
        # Standard forward pass (for validation/Teacher Forcing)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        residual = self.fc(last_hidden)
        return residual, None

    def forward_step(self, x_step, hidden_state):
        """
        Runs ONE step of the LSTM.
        x_step: (Batch, 1, InputDim)
        hidden_state: (h, c) tuple
        """
        lstm_out, new_hidden = self.lstm(x_step, hidden_state)
        # Predict delta for this step
        residual = self.fc(lstm_out[:, -1, :])
        return residual, new_hidden

# ----------------------------------------------------------------------------
# 2. Existing Helpers (Kept Same)
# ----------------------------------------------------------------------------
def is_trajectory_stable(delayed_seq: np.ndarray, true_target: np.ndarray) -> bool:
    if np.isnan(delayed_seq).any() or np.isnan(true_target).any(): return False
    if np.max(np.abs(delayed_seq[:, 7:14])) > 6.0: return False
    if np.max(np.abs(delayed_seq[:, 0:7])) > 6.0: return False
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
        self.ptr = 0
        self.size = 0
        self.seq_len = RNN_SEQUENCE_LENGTH
        self.state_dim = N_JOINTS * 2 + 1 
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.state_dim), dtype=np.float32)
        self.true_targets = np.zeros((buffer_size, N_JOINTS * 2), dtype=np.float32)

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

def collect_data_from_envs(env: VecEnv, num_envs: int) -> Tuple[np.ndarray, np.ndarray]:
    delayed_flat_list = env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
    true_target_list = env.env_method("get_true_current_target")
    raw_seqs = np.array([buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2 + 1) for buf in delayed_flat_list])
    raw_targets = np.array(true_target_list)
    valid_seqs, valid_targets = [], []
    for i in range(num_envs):
        if is_trajectory_stable(raw_seqs[i], raw_targets[i]):
            valid_seqs.append(raw_seqs[i])
            valid_targets.append(raw_targets[i])
    if len(valid_seqs) == 0: return np.array([]), np.array([])
    return np.array(valid_seqs), np.array(valid_targets)

# ----------------------------------------------------------------------------
# 3. Main Training Logic with Scheduled Sampling
# ----------------------------------------------------------------------------

def train_autoregressive_estimator(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Autoregressive_LSTM_{args.config.name}_{timestamp}"
    output_dir = os.path.join(CHECKPOINT_DIR_LSTM, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)    
    logger.info("="*80)
    logger.info("AUTOREGRESSIVE LSTM TRAINING (Closed-Loop)")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    # Env Setup
    def make_env(rank: int):
        def _init():
            return TeleoperationEnvWithDelay(
                delay_config=args.config,
                trajectory_type=args.trajectory_type,
                randomize_trajectory=args.randomize_trajectory,
                seed=args.seed + rank
            )
        return _init
    
    n_envs = NUM_ENVIRONMENTS
    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    replay_buffer = ReplayBuffer(ESTIMATOR_BUFFER_SIZE, device)
    
    # Model
    model = AutoregressiveStateEstimator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ESTIMATOR_LEARNING_RATE)
    
    logger.info("Collecting Initial Data...")
    train_env.reset()
    for _ in range(100): train_env.step([np.zeros((n_envs, N_JOINTS))]) # Warmup
    
    collected = 0
    while collected < ESTIMATOR_BUFFER_SIZE // 2: # Start with half buffer
        seqs, targets = collect_data_from_envs(train_env, n_envs)
        if len(seqs) > 0:
            for i in range(len(seqs)): replay_buffer.add(seqs[i], targets[i])
            collected += len(seqs)
            train_env.step([np.zeros((n_envs, N_JOINTS))])
        else:
            train_env.reset()
            for _ in range(50): train_env.step([np.zeros((n_envs, N_JOINTS))])

    logger.info(f"Buffer filled. Starting Training Loop.")

    # --- Training Loop ---
    model.train()
    
    for update in range(ESTIMATOR_TOTAL_UPDATES):
        
        # 1. Update Teacher Forcing Ratio (Decay over time)
        # Starts at 1.0 (All Real), decays to 0.0 (All Autoregressive) over time
        # We clamp it at 0.3 to ensure it still sees some reality
        progress = update / ESTIMATOR_TOTAL_UPDATES
        teacher_forcing_ratio = max(0.3, 1.0 - (progress * 1.5)) 
        
        # 2. Collect New Data continuously
        seqs, targets = collect_data_from_envs(train_env, n_envs)
        if len(seqs) > 0:
            for i in range(len(seqs)): replay_buffer.add(seqs[i], targets[i])
        train_env.step([np.zeros((n_envs, N_JOINTS))])
        
        # 3. Sample Batch
        batch = replay_buffer.sample(ESTIMATOR_BATCH_SIZE)
        input_seq = batch['delayed_sequences'] # (B, Seq, 15)
        true_final_target = batch['true_targets'] # (B, 14)
        
        # 4. Determine "Cutoff" point for this batch
        # We will simulate packet loss for the last K steps
        # K is random between 1 and 10 steps
        packet_loss_steps = np.random.randint(1, 15)
        cutoff_idx = RNN_SEQUENCE_LENGTH - packet_loss_steps
        
        # Split input into "Safe History" and "Future to Hallucinate"
        safe_history = input_seq[:, :cutoff_idx, :] # (B, T-K, 15)
        
        # --- A. Process Safe History ---
        # Run LSTM over the safe part to get the hidden state
        # We don't care about outputs here, just the state at cutoff
        _, hidden_state = model.lstm(safe_history)
        
        # Initial input for recursion is the last safe frame
        current_input = safe_history[:, -1:, :] # (B, 1, 15)
        
        # --- B. Recursive Hallucination Loop ---
        for k in range(packet_loss_steps):
            
            # Predict Delta based on current_input and hidden_state
            pred_delta, hidden_state = model.forward_step(current_input, hidden_state)
            
            # Reconstruct the "Predicted State"
            # State = Input_Pos + Delta
            # current_input: [B, 1, 15]. Slice first 14 dims (q, qd)
            last_known_state = current_input[:, :, :14] 
            predicted_next_state = last_known_state + (pred_delta.unsqueeze(1) * TARGET_DELTA_SCALE)
            
            # SCHEDULED SAMPLING LOGIC:
            # Should we feed the "Real" next frame (if we had it) or the "Predicted" one?
            # Note: In this specific "packet loss" training, we assume we DON'T have real data 
            # for the dropped packets. So we MUST use prediction + incremented delay.
            
            # Construct next input: [Predicted_State (14) + Delay (1)]
            # Increment delay by DT per step
            current_delay_norm = current_input[:, :, 14] # Get delay from input
            next_delay_norm = current_delay_norm + (DT / DELAY_INPUT_NORM_FACTOR)
            
            # Assemble the input for the next loop iteration
            next_input = torch.cat([predicted_next_state, next_delay_norm.unsqueeze(2)], dim=2)
            
            current_input = next_input # Set for next loop
            
        # --- C. Final Loss Calculation ---
        # Compare the final predicted state (after K steps of drift) to the True Target
        # Note: current_input contains the State at T_final
        final_predicted_state = current_input[:, 0, :14]
        
        # We compare directly against true target (Absolute Error) 
        # because the 'delta' target in buffer assumes 1 step, but we did K steps.
        loss = F.l1_loss(final_predicted_state, true_final_target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if update % 100 == 0:
            tb_writer.add_scalar('loss/train', loss.item(), update)
            tb_writer.add_scalar('training/teacher_forcing_ratio', teacher_forcing_ratio, update)
            tb_writer.add_scalar('training/packet_loss_steps', packet_loss_steps, update)
            print(f"Update {update} | Loss: {loss.item():.6f} | Packet Loss Sim: {packet_loss_steps} steps")
            
    # Save
    torch.save({'state_estimator_state_dict': model.state_dict()}, os.path.join(output_dir, "autoregressive_estimator.pth"))
    logger.info("Training Complete.")
    train_env.close()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Autoregressive LSTM")
    parser.add_argument("--config", type=str, default="3", choices=['1','2','3','4'])
    parser.add_argument("--trajectory-type", type=str, default="figure_8", choices=[t.value for t in TrajectoryType])
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--randomize-trajectory", action="store_true", help="Enable trajectory randomization")
    args = parser.parse_args()

    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1],
                  '3': config_options[2], '4': config_options[3]}
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    return args

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_arguments()
    train_autoregressive_estimator(args)