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
import multiprocessing

from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig, DelaySimulator
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg


def setup_logging(output_dir: str) -> logging.Logger:
    log_file = os.path.join(output_dir, "autoregressive_train.log")
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file), 
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.seq_len = cfg.RNN_SEQUENCE_LENGTH
        
        # Explicit dimensions
        self.input_dim = cfg.ESTIMATOR_STATE_DIM # 15
        self.target_dim = cfg.ESTIMATOR_OUTPUT_DIM # 14 (Single step)
      
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.input_dim), dtype=np.float32)
        self.target_single = np.zeros((buffer_size, self.target_dim), dtype=np.float32)
        
    def add(self, delayed_seq: np.ndarray, target_single: np.ndarray) -> None:
        self.delayed_sequences[self.ptr] = delayed_seq
        self.target_single[self.ptr] = target_single
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'delayed_sequences': torch.tensor(self.delayed_sequences[indices], device=self.device),
            'target_single': torch.tensor(self.target_single[indices], device=self.device),
        }
        
    def __len__(self): return self.size

def collect_data_from_envs(env: SubprocVecEnv, num_envs: int):
    """
    Collects a batch of (Past Sequence, Next Step Target) pairs.
    """
    INPUT_DIM = cfg.ESTIMATOR_STATE_DIM
    
    # 1. Get Past (Delayed Sequences)
    delayed_flat_list = env.env_method("get_delayed_target_buffer", cfg.RNN_SEQUENCE_LENGTH)
    
    # 2. Get Future (Single Step Ground Truth)
    true_single_list = env.env_method("get_future_target_single") 
    
    # Reshape using explicit dimensions
    raw_seqs = np.array([buf.reshape(cfg.RNN_SEQUENCE_LENGTH, INPUT_DIM) for buf in delayed_flat_list])
    raw_targets = np.array(true_single_list) 
    
    valid_seqs, valid_targets = [], []
    for i in range(num_envs):
        # Basic NaN check
        if not np.isnan(raw_seqs[i]).any() and not np.isnan(raw_targets[i]).any():
            valid_seqs.append(raw_seqs[i])
            valid_targets.append(raw_targets[i])
            
    if len(valid_seqs) == 0: return np.array([]), np.array([])
    return np.array(valid_seqs), np.array(valid_targets)

def validate(model: nn.Module, val_env: SubprocVecEnv, device: torch.device, num_samples: int = 500) -> float:
    model.eval()
    total_val_loss = 0.0
    samples_count = 0
    
    val_env.reset()
    
    with torch.no_grad():
        while samples_count < num_samples:
            val_env.step([np.zeros((val_env.num_envs, cfg.N_JOINTS))])
            
            seqs, targets = collect_data_from_envs(val_env, val_env.num_envs)
            if len(seqs) == 0: continue
                
            input_seq = torch.tensor(seqs, dtype=torch.float32).to(device)
            target_single = torch.tensor(targets, dtype=torch.float32).to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred_state, _ = model(input_seq)
                loss = F.mse_loss(pred_state, target_single)
            
            total_val_loss += loss.item() * len(seqs)
            samples_count += len(seqs)
            
    model.train()
    return total_val_loss / samples_count if samples_count > 0 else 0.0

def train(args):
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"LSTM_Step_{args.config.name}_{timestamp}"
    output_dir = os.path.join(cfg.CHECKPOINT_DIR_LSTM, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)    
    
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"TensorBoard logs will be saved to: {tb_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device} (GTX 1650 Optimized)")
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    def make_env_fn(rank: int, config: ExperimentConfig, traj_type: TrajectoryType):
        def _init():
            return TeleoperationEnvWithDelay(
                delay_config=config,
                trajectory_type=traj_type,
                randomize_trajectory=args.randomize_trajectory,
                seed=args.seed + rank,
                lstm_model_path=None 
            )
        return _init
    
    # 1. Training Environment
    train_env = SubprocVecEnv([
        make_env_fn(i, args.config, args.trajectory_type) 
        for i in range(cfg.NUM_ENVIRONMENTS)
    ])
    
    # 2. Validation Environment
    val_traj_type = args.trajectory_type
    val_env = SubprocVecEnv([
        make_env_fn(10000, args.config, val_traj_type) 
        for _ in range(1)
    ])
    
    logger.info(f"Training Trajectory:   {args.trajectory_type.value}")
    logger.info(f"Validation Trajectory: {val_traj_type.value} (Same Type, Different Seed)")
    
    # --- Model & Optimizer ---
    # Using 14D output dim (Step-Based)
    model = StateEstimator(output_dim=cfg.N_JOINTS * 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ESTIMATOR_LEARNING_RATE)
    replay_buffer = ReplayBuffer(cfg.ESTIMATOR_BUFFER_SIZE, device)
    
    # --- Buffer Filling ---
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
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for update in range(1, cfg.ESTIMATOR_TOTAL_UPDATES + 1):
        seqs, targets = collect_data_from_envs(train_env, cfg.NUM_ENVIRONMENTS)
        if len(seqs) > 0:
            for i in range(len(seqs)): replay_buffer.add(seqs[i], targets[i])
        train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])
        
        batch = replay_buffer.sample(cfg.ESTIMATOR_BATCH_SIZE)
        input_seq = batch['delayed_sequences'] 
        target_single = batch['target_single']    
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred_state, _ = model(input_seq)
            loss = F.mse_loss(pred_state, target_single)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        tb_writer.add_scalar("Train/Loss", loss.item(), update)
        
        if update % cfg.ESTIMATOR_VAL_FREQ == 0:
            val_loss = validate(model, val_env, device)
            
            logger.info(f"Update {update} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}")
            tb_writer.add_scalar("Val/Loss", val_loss, update)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0 
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                logger.info(f"  [>] New Best Model Saved (Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                logger.info(f"  [!] No improvement. Patience: {patience_counter}/{cfg.ESTIMATOR_PATIENCE}")
                
                if patience_counter >= cfg.ESTIMATOR_PATIENCE:
                    logger.info("Early stopping triggered.")
                    break

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    
    tb_writer.close()
    train_env.close()
    val_env.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="4")
    parser.add_argument("--trajectory-type", type=str, default="figure_8")
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
    train(parse_arguments())