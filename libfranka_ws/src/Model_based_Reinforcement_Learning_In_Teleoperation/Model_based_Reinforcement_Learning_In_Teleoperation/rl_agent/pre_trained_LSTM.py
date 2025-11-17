"""
Pre-training script for the State Estimator (LSTM).

MODIFIED: This script will now save the warmup data to a CSV and exit.

Test Data collection

pipelines:
1. Collect data by running the TeleoperationEnvWithDelay environment with a random policy.
2. Store (delayed_sequence, true_target) pairs in a replay buffer.
3. <<< SAVE BUFFER TO CSV AND EXIT >>>
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
import pandas as pd # <<< MODIFICATION: Import pandas

# Stable Baselines3 imports
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# PyTorch imports
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Learning rate scheduler

# <<< MODIFICATION: Ensure all necessary components are imported >>>
try:
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
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure this script is run from a directory where modules are accessible.")
    sys.exit(1)


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
    logger.info("LSTM STATE ESTIMATOR PRE-TRAINING (DATA COLLECTION ONLY)") # <<< MODIFICATION
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
    
    # Initialize replay buffers
    replay_buffer = PretrainReplayBuffer(ESTIMATOR_BUFFER_SIZE, device)
    
    # Filling training buffer 
    logger.info("Filling training buffer...")
    obs = train_env.reset()
    for step in range(ESTIMATOR_WARMUP_STEPS):
        delayed_seq_batch, true_target_batch = collect_data_from_envs(train_env, NUM_ENVIRONMENTS)
        for i in range(NUM_ENVIRONMENTS):
            replay_buffer.add(delayed_seq_batch[i], true_target_batch[i])
        
        random_actions = np.array([train_env.action_space.sample() for _ in range(NUM_ENVIRONMENTS)])
        obs, rewards, dones, infos = train_env.step(random_actions)
        
        if step % 1000 == 0:
            logger.info(f"Collection step {step}/{ESTIMATOR_WARMUP_STEPS}...")
            
    logger.info(f"Training buffer filled: {len(replay_buffer)} samples")
    train_env.close()

    # <<< MODIFICATION: Save buffer to CSV and exit >>>
    logger.info("Saving collected data to CSV...")
    
    try:
        # Get all collected data from the buffer
        n_samples = replay_buffer.size
        state_dim = replay_buffer.state_dim
        seq_len = replay_buffer.seq_len
        
        # Get data, ensuring we only take what was filled
        delayed_seq_data = replay_buffer.delayed_sequences[:n_samples]
        true_target_data = replay_buffer.true_targets[:n_samples]
        
        # Flatten the sequence data: (n_samples, seq_len, state_dim) -> (n_samples, seq_len * state_dim)
        flattened_seq = delayed_seq_data.reshape(n_samples, seq_len * state_dim)
        
        # Create column names
        # Create descriptive column names
        q_qd = ['q', 'qd']
        seq_cols = [
            f"seq_t{t}_j{j}_{v}" 
            for t in range(seq_len) 
            for j in range(N_JOINTS) 
            for v in q_qd
        ]
        target_cols = [
            f"target_j{j}_{v}" 
            for j in range(N_JOINTS) 
            for v in q_qd
        ]
        
        # Create pandas DataFrames
        df_seq = pd.DataFrame(flattened_seq, columns=seq_cols)
        df_target = pd.DataFrame(true_target_data, columns=target_cols)
        
        # Combine into one large DataFrame
        df_combined = pd.concat([df_seq, df_target], axis=1)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "collected_warmup_data.csv")
        df_combined.to_csv(csv_path, index=False)
        
        logger.info("="*80)
        logger.info(f"Successfully saved {n_samples} data points to:")
        logger.info(f"{csv_path}")
        logger.info("Script will now exit.")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Failed to save data to CSV: {e}")
        
    finally:
        # Exit script regardless of save success
        sys.exit(0)
    # <<< END OF MODIFICATION >>>


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-train the State Estimator LSTM for teleoperation with delays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config", type=str, default="3", choices=['1', '2', '3', '4'], help="Delay configuration preset.")
    parser.add_argument("--trajectory-type", type=str, default="figure_8", choices=[t.value for t in TrajectoryType], help="Reference trajectory type.")
    parser.add_argument("--seed", type=int, default=50, help="Random seed.")
    
    args = parser.parse_args()
    
    # Convert string arguments to enum types
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {
        '1': config_options[0], 
        '2': config_options[1],
        '3': config_options[2], 
        '4': config_options[3]
    }
    args.config = CONFIG_MAP[args.config]
    args.trajectory_type = next(
        t for t in TrajectoryType 
        if t.value.lower() == args.trajectory_type.lower()
    )
    
    return args


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    args = parse_arguments()
    
    try:
        pretrain_estimator(args)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise