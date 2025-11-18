"""
Pre-training script for the State Estimator (LSTM).

pipelines:
1. Collect data by running the TeleoperationEnvWithDelay environment with a random policy.
2. Store (delayed_sequence, true_target) pairs in a replay buffer.
3. Train the StateEstimator LSTM in a supervised learning. (min MSE loss)
"""

# Python imports 
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

# Stable Baselines3 imports
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# PyTorch imports
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Learning rate scheduler

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
    ESTIMATOR_WARMUP_STEPS,
    ESTIMATOR_TOTAL_UPDATES,
    NUM_ENVIRONMENTS,
    CHECKPOINT_DIR_LSTM,
    ESTIMATOR_VAL_STEPS,
    ESTIMATOR_VAL_FREQ,
    ESTIMATOR_PATIENCE,
    ESTIMATOR_LR_PATIENCE,
    TRAJECTORY_RANDOM,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
)


class PretrainReplayBuffer:
    """Replay buffer for storing delayed observation sequences and corresponding true states."""
    
    def __init__(self, buffer_size: int, device: torch.device):
        
        # Initialize parameters
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0  # Initial pointer
        self.size = 0
        
        self.seq_len = RNN_SEQUENCE_LENGTH
        self.state_dim = N_JOINTS * 2  # target_q and target_qd
        
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.state_dim), dtype=np.float32) # dim: (buffer_size, seq_len, state_dim)
        self.true_targets = np.zeros((buffer_size, self.state_dim), dtype=np.float32) # dim: (buffer_size, state_dim)

    def add(self, delayed_seq: np.ndarray, true_target: np.ndarray) -> None:
        """Add a single (delayed_sequence, true_target) pair to the buffer."""
        self.delayed_sequences[self.ptr] = delayed_seq  # Training data
        self.true_targets[self.ptr] = true_target  # Groundtruth data
        # The pointer is to match the positions in both buffers
        
        self.ptr = (self.ptr + 1) % self.buffer_size  # pointer moves to the next slot
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch from the buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'delayed_sequences': torch.tensor(self.delayed_sequences[indices], device=self.device),
            'true_targets': torch.tensor(self.true_targets[indices], device=self.device),
        }
        return batch

    def __len__(self) -> int:
        return self.size


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to file and console."""
    log_file = os.path.join(output_dir, "pretrain_estimator.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def collect_data_from_envs(env: VecEnv, num_envs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect delayed sequences and true targets from all environments.
    
    Returns:
        delayed_seq_batch: shape (num_envs, seq_len, state_dim)
        true_target_batch: shape (num_envs, state_dim)
    """
    delayed_buffers_list = env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
    true_targets_list = env.env_method("get_true_current_target")
    
    delayed_seq_batch = np.array([
        buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2) 
        for buf in delayed_buffers_list
    ])
    true_target_batch = np.array(true_targets_list)
    
    return delayed_seq_batch, true_target_batch


def evaluate_model(model: StateEstimator, val_buffer: PretrainReplayBuffer, batch_size: int, num_batches: int = 50) -> float:
    """
    Calculate the average loss on the validation buffer.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    
    if len(val_buffer) < batch_size:
        return float('inf') # Not enough data to validate

    with torch.no_grad():  # Disable gradient calculation
        for _ in range(num_batches):
            batch = val_buffer.sample(batch_size)
            predicted_targets, _ = model(batch['delayed_sequences'])
            loss = F.mse_loss(predicted_targets, batch['true_targets'])
            total_loss += loss.item()
            
    model.train()  # Set model back to training mode
    return total_loss / num_batches


def pretrain_estimator(args: argparse.Namespace) -> None:
    """Main pre-training function for the State Estimator LSTM."""
    
    # Setup output directory and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Pretrain_LSTM_{args.config.name}_{timestamp}"
    output_dir = os.path.join(CHECKPOINT_DIR_LSTM, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    logger.info("="*80)
    logger.info("LSTM STATE ESTIMATOR PRE-TRAINING")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Delay Config: {args.config.name}")
    logger.info(f"  Train Trajectory: {args.trajectory_type.value}")
    logger.info(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"  Environments: {NUM_ENVIRONMENTS}")
    logger.info(f"  Sequence Length: {RNN_SEQUENCE_LENGTH}")
    logger.info(f"Hyperparameters:")
    logger.info(f"  Learning Rate: {ESTIMATOR_LEARNING_RATE}")
    logger.info(f"  Batch Size: {ESTIMATOR_BATCH_SIZE}")
    logger.info(f"  Buffer Size: {ESTIMATOR_BUFFER_SIZE}")
    logger.info(f"  Total Updates: {ESTIMATOR_TOTAL_UPDATES}")
    logger.info(f"  Warmup Steps: {ESTIMATOR_WARMUP_STEPS}")
    logger.info(f"  Validation Frequency: {ESTIMATOR_VAL_FREQ} steps")
    logger.info(f"  Early Stopping Patience: {ESTIMATOR_PATIENCE} checks")
    logger.info(f"  LR Scheduler Patience: {ESTIMATOR_LR_PATIENCE} checks")
    logger.info(f"  Trajectory Randomization during Training: {TRAJECTORY_RANDOM}")
    logger.info(f"  RNN Hidden Dimension: {RNN_HIDDEN_DIM}")
    logger.info(f"  RNN Number of Layers: {RNN_NUM_LAYERS}")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    
    # Create training environment
    def make_train_env():
        return TeleoperationEnvWithDelay(
            delay_config=args.config,
            trajectory_type=args.trajectory_type,
            randomize_trajectory=TRAJECTORY_RANDOM,
            seed=args.seed
        )
    
    train_env = make_vec_env(
        make_train_env,
        n_envs=NUM_ENVIRONMENTS,
        seed=args.seed,
        vec_env_cls= SubprocVecEnv if NUM_ENVIRONMENTS > 1 else DummyVecEnv
    )
    replay_buffer = PretrainReplayBuffer(ESTIMATOR_BUFFER_SIZE, device)
    
    if args.overfit:
        logger.info("OVERFIT MODE: Collecting full clean trajectory offline...")
        obs = train_env.reset()
        
        # Throw away first 1000 steps (warmup + stable start)
        for _ in range(1000):
            train_env.step([train_env.action_space.sample()])  # random or zero – doesn't matter
        
        collected = 0
        max_collect = 80000  # ~80 seconds of data → more than enough
        
        while collected < max_collect:
            delayed_seq_batch, true_target_batch = collect_data_from_envs(train_env, 1)
            
            replay_buffer.add(delayed_seq_batch[0], true_target_batch[0])
            collected += 1
            
            # Step with zero torque → follower doesn't interfere
            action = np.zeros((1, N_JOINTS), dtype=np.float32)
            obs, _, _, _ = train_env.step([action])
            
            if collected % 5000 == 0:
                logger.info(f"  Collected {collected}/{max_collect} clean samples...")

        logger.info(f"OFFLINE COLLECTION DONE → buffer size = {len(replay_buffer)}")
    else:
        logger.info("Standard online collection (not recommended until overfit works)")

    # Validation buffer stays the same (still online, but short)
    val_env = make_vec_env(make_train_env, n_envs=1, vec_env_cls=DummyVecEnv)
    val_buffer = PretrainReplayBuffer(ESTIMATOR_VAL_STEPS, device)
    logger.info("Filling validation buffer...")
    val_obs = val_env.reset()
    for _ in range(ESTIMATOR_VAL_STEPS):
        delayed_seq, true_target = collect_data_from_envs(val_env, 1)
        val_buffer.add(delayed_seq[0], true_target[0])
        action = np.zeros((1, N_JOINTS), dtype=np.float32)
        val_obs, _, _, _ = val_env.step([action])
    val_env.close()

    # Model & optimizer
    state_estimator = StateEstimator().to(device)
    optimizer = torch.optim.Adam(state_estimator.parameters(), lr=ESTIMATOR_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=ESTIMATOR_LR_PATIENCE, verbose=True)

    state_estimator.train()
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history = deque(maxlen=100)

    logger.info("Starting training loop...")
    
    for update in range(ESTIMATOR_TOTAL_UPDATES):
        
        # MODIFICATION: In overfit mode we DO NOT collect online anymore
        if not args.overfit and update > 500:  # old buggy behaviour – kept only for non-overfit
            delayed_seq_batch, true_target_batch = collect_data_from_envs(train_env, 1)
            replay_buffer.add(delayed_seq_batch[0], true_target_batch[0])
            action = np.zeros((1, N_JOINTS), dtype=np.float32)
            train_env.step([action])

        # Training step (same)
        batch = replay_buffer.sample(ESTIMATOR_BATCH_SIZE)
        predicted_targets, _ = state_estimator(batch['delayed_sequences'])
        loss = F.mse_loss(predicted_targets, batch['true_targets'])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state_estimator.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(loss.item())

        # Validation
        if update % ESTIMATOR_VAL_FREQ == 0:
            avg_train_loss = np.mean(loss_history)
            val_loss = evaluate_model(state_estimator, val_buffer, ESTIMATOR_BATCH_SIZE)
            
            tb_writer.add_scalar('loss/train_avg_100', avg_train_loss, update)
            tb_writer.add_scalar('loss/validation_mse', val_loss, update)
            tb_writer.add_scalar('hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], update)
            
            logger.info(f"Update {update:05d} | Train {avg_train_loss:.8f} | Val {val_loss:.8f}")

            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'state_estimator_state_dict': state_estimator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'update': update,
                }, os.path.join(output_dir, "estimator_best.pth"))
                logger.info("  -> New best! Saved.")
            else:
                patience_counter += 1
                if patience_counter >= ESTIMATOR_PATIENCE:
                    logger.info(f"Early stopping at update {update}")
                    break

    # Final save & test
    torch.save({
        'state_estimator_state_dict': state_estimator.state_dict(),
        'best_val_loss': best_val_loss,
    }, os.path.join(output_dir, "estimator_final.pth"))

    logger.info(f"Training finished! Best Val MSE = {best_val_loss:.8f}")
    logger.info(f"Model saved to {output_dir}")
    tb_writer.close()
    train_env.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train the State Estimator LSTM (fixed + overfit mode)")
    
    parser.add_argument("--config", type=str, default="1", choices=['1', '2', '3', '4'],
                        help="Delay configuration (use '1' first for debugging)")
    parser.add_argument("--trajectory-type", type=str, default="figure_8",
                        choices=[t.value for t in TrajectoryType])
    parser.add_argument("--seed", type=int, default=50)
    # MODIFICATION: New flags
    parser.add_argument("--overfit", action="store_true",
                        help="Enable overfit mode: collect full clean trajectory offline (RECOMMENDED for debugging)")
    parser.add_argument("--randomize-trajectory", action="store_true",
                        help="Randomize trajectory params (only use after overfit works)")

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
    
    # In overfit mode we FORCE deterministic + no randomisation
    if args.overfit:
        args.randomize_trajectory = False
        print("OVERFIT MODE ACTIVATED – forcing deterministic trajectory")

    pretrain_estimator(args)