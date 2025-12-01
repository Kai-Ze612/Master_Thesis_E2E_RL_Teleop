"""
Autoregressive LSTM state estimator training script.

Features:
1. Early fusion LSTM:
    - Inputs: Sequence of delayed observations + normalized delay value (15D: 7D q, 7D qd, 1D delay)
    - Delay is encoded because the LSTM needs to know how far back in time the last observation was.
2. Autoregressive prediction:
    - During packet loss simulation(no new input period), the model predicts multiple steps ahead by feeding its own predictions back
    - If there is new data, it resets the hidden state with the new observation, so that the data won't be polluted by old predictions.
    - Risk: When the delay is very large, the model has to predict many steps ahead, which can lead to divergence.
3. 10 Steps Ahead Prediction:
    - The model is trained to predict 10 steps ahead in one forward pass.
    - This allows to lower the inference time
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
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig, DelaySimulator
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import TrajectoryType

import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg


class AutoregressiveStateEstimator(nn.Module):
    def __init__(self):
        """
        input_dim: 14D(q + qd) + 1D (normalized delay)
        output_dim: 14D (predicted q + qd) * shot_size (10)
        """
        
        super().__init__()

        self.shot_size = cfg.ESTIMATOR_PREDICTION_HORIZON # 10 steps
        self.state_dim = cfg.ESTIMATOR_OUTPUT_DIM  # 14 (q + qd)

        self.lstm = nn.LSTM(
            input_size=cfg.ESTIMATOR_STATE_DIM,
            hidden_size=cfg.RNN_HIDDEN_DIM, 
            num_layers=cfg.RNN_NUM_LAYERS,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(cfg.RNN_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_dim * self.shot_size) 
        )

    def forward(self, x):
        """
        Training forward pass.
        Input: Sequence of delayed observations (Batch, Seq_Len, 15)
        Output: Prediction shot (Batch, 10, 14)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Predict flattened shot (Batch, 140)
        flat_shot = self.fc(last_hidden)
        
        # Reshape to (Batch, 10, 14)
        pred_shot = flat_shot.view(-1, self.shot_size, self.state_dim)
        
        return pred_shot, None

    def forward_shot(self, x_step, hidden_state):
        """
        Inference forward pass (used by Env).
        Input: Single step input (Batch, 1, 15) and previous hidden state
        Output: Prediction shot (Batch, 10, 14) and new hidden state
        """
        lstm_out, new_hidden = self.lstm(x_step, hidden_state)
        
        # Only take the last step's hidden state
        last_hidden_step = lstm_out[:, -1, :]
        
        flat_shot = self.fc(last_hidden_step)
        pred_shot = flat_shot.view(-1, self.shot_size, self.state_dim)
        
        return pred_shot, new_hidden

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
        self.seq_len = cfg.RNN_SEQUENCE_LENGTH
        self.state_dim = cfg.ESTIMATOR_STATE_DIM 
        self.shot_size = cfg.ESTIMATOR_PREDICTION_HORIZON
      
        # We still store true_targets for reference/validation, though we won't train on them directly
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, cfg.ESTIMATOR_STATE_DIM), dtype=np.float32)  ## training sequences
        self.target_shots = np.zeros((buffer_size, self.shot_size, 14), dtype=np.float32)
        
    def add(self, delayed_seq: np.ndarray, target_shot: np.ndarray) -> None:
        self.delayed_sequences[self.ptr] = delayed_seq
        self.target_shots[self.ptr] = target_shot
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'delayed_sequences': torch.tensor(self.delayed_sequences[indices], device=self.device),
            'target_shots': torch.tensor(self.target_shots[indices], device=self.device),
        }
        
    def __len__(self): return self.size

def collect_data_from_envs(env: SubprocVecEnv, num_envs: int):
    # Get Past
    delayed_flat_list = env.env_method("get_delayed_target_buffer", cfg.RNN_SEQUENCE_LENGTH)
    # Get Future (10 steps)
    true_shot_list = env.env_method("get_future_target_chunk", 10) 
    
    raw_seqs = np.array([buf.reshape(cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM) for buf in delayed_flat_list])
    raw_targets = np.array(true_shot_list) 
    
    valid_seqs, valid_targets = [], []
    for i in range(num_envs):
        # Basic NaN check
        if not np.isnan(raw_seqs[i]).any() and not np.isnan(raw_targets[i]).any():
            valid_seqs.append(raw_seqs[i])
            valid_targets.append(raw_targets[i])
            
    if len(valid_seqs) == 0: return np.array([]), np.array([])
    return np.array(valid_seqs), np.array(valid_targets)

def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"LSTM_Shot10_FP16_{args.config.name}_{timestamp}"
    output_dir = os.path.join(cfg.CHECKPOINT_DIR_LSTM, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device} (GTX 1650 Optimized)")
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    def make_env(rank: int):
        def _init():
            return TeleoperationEnvWithDelay(
                delay_config=args.config,
                trajectory_type=args.trajectory_type,
                randomize_trajectory=args.randomize_trajectory,
                seed=args.seed + rank,
                lstm_model_path=None 
            )
        return _init
    
    train_env = SubprocVecEnv([make_env(i) for i in range(cfg.NUM_ENVIRONMENTS)])
    
    replay_buffer = ReplayBuffer(cfg.ESTIMATOR_BUFFER_SIZE, device)
    model = AutoregressiveStateEstimator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ESTIMATOR_LEARNING_RATE)
    
    # Fill Buffer
    logger.info("Filling buffer...")
    train_env.reset()
    collected = 0
    while collected < cfg.ESTIMATOR_BUFFER_SIZE // 5:
        seqs, targets = collect_data_from_envs(train_env, cfg.NUM_ENVIRONMENTS)
        if len(seqs) > 0:
            for i in range(len(seqs)): replay_buffer.add(seqs[i], targets[i])
            collected += len(seqs)
        train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])

    logger.info("Training...")
    model.train()
    
    for update in range(1, cfg.ESTIMATOR_TOTAL_UPDATES + 1):
        seqs, targets = collect_data_from_envs(train_env, cfg.NUM_ENVIRONMENTS)
        if len(seqs) > 0:
            for i in range(len(seqs)): replay_buffer.add(seqs[i], targets[i])
        train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])
        
        batch = replay_buffer.sample(cfg.ESTIMATOR_BATCH_SIZE)
        input_seq = batch['delayed_sequences'] 
        target_shot = batch['target_shots']    
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred_shot, _ = model(input_seq)
            loss = F.mse_loss(pred_shot, target_shot)
        
        # Scale loss to prevent underflow in FP16
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if update % 100 == 0:
            print(f"Update {update} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    train_env.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="4")
    parser.add_argument("--trajectory-type", type=str, default="figure_8")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--randomize-trajectory", action="store_true")
    args = parser.parse_args()
    
    # Convert config to ExperimentConfig
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2], '4': config_options[3]}
    args.config = CONFIG_MAP[args.config]
    
    # Trajectory Type
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    
    return args

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    train(parse_arguments())