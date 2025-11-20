"""
Pre-training script for the State Estimator (LSTM).

pipelines:
1. Collect data by running the TeleoperationEnvWithDelay environment with a random policy.
2. Store (delayed_sequence, true_target) pairs in a replay buffer.
3. Train the StateEstimator LSTM in a supervised learning. (min MSE loss)
"""

import os
import sys
import torch
import numpy as np
import argparse
import logging
from datetime import datetime
from collections import deque
import torch.nn.functional as F
from typing import Dict, Tuple
import multiprocessing
import time

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS, 
    RNN_SEQUENCE_LENGTH,
    ESTIMATOR_LEARNING_RATE,
    ESTIMATOR_BATCH_SIZE,
    ESTIMATOR_BUFFER_SIZE,
    ESTIMATOR_TOTAL_UPDATES,
    NUM_ENVIRONMENTS,
    CHECKPOINT_DIR_LSTM,
    ESTIMATOR_VAL_STEPS,
    ESTIMATOR_VAL_FREQ,
    ESTIMATOR_PATIENCE,
    ESTIMATOR_LR_PATIENCE,
    INITIAL_JOINT_CONFIG,
)

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

    def __len__(self) -> int:
        return self.size

def setup_logging(output_dir: str) -> logging.Logger:
    log_file = os.path.join(output_dir, "pretrain_estimator.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

def collect_data_from_envs(env: VecEnv, num_envs: int) -> Tuple[np.ndarray, np.ndarray]:
    delayed_flat_list = env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
    true_target_list = env.env_method("get_true_current_target")
    
    delayed_seq_batch = np.array([
        buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2 + 1)
        for buf in delayed_flat_list
    ])
    true_target_batch = np.array(true_target_list)
    return delayed_seq_batch, true_target_batch

def evaluate_model(model: StateEstimator, val_buffer: ReplayBuffer, batch_size: int, num_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    if len(val_buffer) < batch_size: return float('inf')
    with torch.no_grad():
        for _ in range(num_batches):
            batch = val_buffer.sample(batch_size)
            pred, _ = model(batch['delayed_sequences'])
            loss = F.mse_loss(pred, batch['true_targets'])
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches

def inject_static_samples(buffer: ReplayBuffer, num_samples: int, logger: logging.Logger):
    """Helper to inject static hold sequences."""
    
    static_input_state = np.concatenate([INITIAL_JOINT_CONFIG, np.zeros(N_JOINTS), [0.0]]).astype(np.float32)
    # Static target: [q, 0]
    static_target_state = np.concatenate([INITIAL_JOINT_CONFIG, np.zeros(N_JOINTS)]).astype(np.float32)
    static_seq = np.tile(static_input_state, (RNN_SEQUENCE_LENGTH, 1))
    for _ in range(num_samples):
        buffer.add(static_seq, static_target_state)
    logger.info(f"Injected {num_samples} static hold samples (Augmented).")

def verify_buffer_coverage(buffer: ReplayBuffer, logger: logging.Logger):
    if buffer.size == 0:
        logger.warning("Buffer is empty! Cannot verify coverage.")
        return
    targets = buffer.true_targets[:buffer.size] 
    positions = targets[:, :N_JOINTS]
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    mean_pos = np.mean(positions, axis=0)
    logger.info("\n" + "="*40)
    logger.info(" BUFFER COVERAGE CHECK (Joint Positions)")
    logger.info("="*40)
    logger.info(f"{'Joint':<5} | {'Min':<8} | {'Max':<8} | {'Range':<8} | {'Mean':<8}")
    logger.info("-" * 45)
    for i in range(N_JOINTS):
        p_range = max_pos[i] - min_pos[i]
        logger.info(f"J{i:<4} | {min_pos[i]:<8.3f} | {max_pos[i]:<8.3f} | {p_range:<8.3f} | {mean_pos[i]:<8.3f}")
        if p_range < 0.01:
            logger.warning(f"  [WARNING] Joint {i} range is very small! Check trajectory generator.")
    logger.info("="*40 + "\n")

def pretrain_estimator(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Pretrain_LSTM_{args.config.name}_{timestamp}"
    output_dir = os.path.join(CHECKPOINT_DIR_LSTM, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)    
    logger.info("="*80)
    logger.info("LSTM STATE ESTIMATOR PRE-TRAINING (Balanced + Delay Augmented)")
    logger.info("="*80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    def make_env(rank: int):
        def _init():
            return TeleoperationEnvWithDelay(delay_config=args.config, trajectory_type=args.trajectory_type, randomize_trajectory=args.randomize_trajectory, seed=args.seed + rank)
        return _init
    
    n_envs = 1 if args.randomize_trajectory == False else NUM_ENVIRONMENTS
    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    train_env = vec_cls([make_env(i) for i in range(n_envs)])
    replay_buffer = ReplayBuffer(ESTIMATOR_BUFFER_SIZE, device)
    num_static = int(ESTIMATOR_BUFFER_SIZE * 0.2)
    num_dynamic = ESTIMATOR_BUFFER_SIZE - num_static
    logger.info(f"Targeting Dataset Balance: {num_static} Static / {num_dynamic} Dynamic")
    inject_static_samples(replay_buffer, num_static, logger)
    logger.info("Collecting dynamic trajectories...")
    train_env.reset()
    for _ in range(100):
        train_env.step([np.zeros((n_envs, N_JOINTS))])

    collected = 0
    while collected < num_dynamic:
        seqs, targets = collect_data_from_envs(train_env, n_envs)
        for i in range(n_envs):
            replay_buffer.add(seqs[i], targets[i])
        collected += n_envs
        train_env.step([np.zeros((n_envs, N_JOINTS))])
        if collected % 5000 < n_envs:
            logger.info(f"  -> {collected}/{num_dynamic} dynamic samples")
            
    logger.info(f"Data collection complete. Buffer size: {len(replay_buffer)}")
    verify_buffer_coverage(replay_buffer, logger)
    val_env = DummyVecEnv([make_env(0)])
    val_buffer = ReplayBuffer(ESTIMATOR_VAL_STEPS, device)
    val_static = int(ESTIMATOR_VAL_STEPS * 0.2)
    val_dynamic = ESTIMATOR_VAL_STEPS - val_static
    logger.info("Filling Validation Buffer (Balanced)...")
    inject_static_samples(val_buffer, val_static, logger)
    val_env.reset()
    for _ in range(100):
         val_env.step([np.zeros((1, N_JOINTS))])
    for _ in range(val_dynamic):
        seq, target = collect_data_from_envs(val_env, 1)
        val_buffer.add(seq[0], target[0])
        val_env.step([np.zeros((1, N_JOINTS))])
    val_env.close()
    
    state_estimator = StateEstimator().to(device)
    optimizer = torch.optim.Adam(state_estimator.parameters(), lr=ESTIMATOR_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=ESTIMATOR_LR_PATIENCE)
    state_estimator.train()
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history = deque(maxlen=100)
    logger.info(f"Starting training...")
    
    for update in range(ESTIMATOR_TOTAL_UPDATES):
        if args.randomize_trajectory:
            delayed_seq_batch, true_target_batch = collect_data_from_envs(train_env, n_envs)
            for i in range(n_envs):
                replay_buffer.add(delayed_seq_batch[i], true_target_batch[i])
            train_env.step([np.zeros((n_envs, N_JOINTS))])

        batch = replay_buffer.sample(ESTIMATOR_BATCH_SIZE)
        pred, _ = state_estimator(batch['delayed_sequences'])
        loss = F.mse_loss(pred, batch['true_targets'])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state_estimator.parameters(), 1.0)
        optimizer.step()
        loss_history.append(loss.item())

        if update % ESTIMATOR_VAL_FREQ == 0:
            train_loss = np.mean(loss_history)
            val_loss = evaluate_model(state_estimator, val_buffer, ESTIMATOR_BATCH_SIZE)
            tb_writer.add_scalar('loss/train', train_loss, update)
            tb_writer.add_scalar('loss/val', val_loss, update)
            logger.info(f"Update {update:05d} | Train {train_loss:.8f} | Val {val_loss:.8f}")
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({'state_estimator_state_dict': state_estimator.state_dict(), 'best_val_loss': best_val_loss, 'update': update}, os.path.join(output_dir, "estimator_best.pth"))
                logger.info("  → NEW BEST!")
            else:
                patience_counter += 1
                if patience_counter >= ESTIMATOR_PATIENCE:
                    logger.info("Early stopping")
                    break

    torch.save({'state_estimator_state_dict': state_estimator.state_dict()}, os.path.join(output_dir, "estimator_final.pth"))
    logger.info(f"Finished. Best Val MSE: {best_val_loss:.8f}")
    tb_writer.close()
    train_env.close()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train LSTM State Estimator")
    parser.add_argument("--config", type=str, default="3", choices=['1','2','3','4'])
    parser.add_argument("--trajectory-type", type=str, default="figure_8", choices=[t.value for t in TrajectoryType])
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--randomize-trajectory", action="store_true", help="Enable trajectory randomization")
    args = parser.parse_args()
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2], '4': config_options[3]}
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    return args

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_arguments()
    pretrain_estimator(args)