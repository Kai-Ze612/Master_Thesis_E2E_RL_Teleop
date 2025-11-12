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

# Stable Baselines3 imports
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from torch.utils.tensorboard import SummaryWriter

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
    DEFAULT_CONTROL_FREQ,
    WARM_UP_DURATION,
)


class PretrainReplayBuffer:
    """Replay buffer for storing delayed observation sequences and corresponding true states."""
    
    def __init__(self, buffer_size: int, device: torch.device):
        
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.seq_len = RNN_SEQUENCE_LENGTH
        self.state_dim = N_JOINTS * 2
        
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.state_dim), dtype=np.float32)
        self.true_targets = np.zeros((buffer_size, self.state_dim), dtype=np.float32)

    def add(self, delayed_seq: np.ndarray, true_target: np.ndarray) -> None:
        """Add a single (delayed_sequence, true_target) pair to the buffer."""
        self.delayed_sequences[self.ptr] = delayed_seq
        self.true_targets[self.ptr] = true_target
        
        self.ptr = (self.ptr + 1) % self.buffer_size
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
    try:
        delayed_buffers_list = env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)
        true_targets_list = env.env_method("get_true_current_target")
        
        delayed_seq_batch = np.array([
            buf.reshape(RNN_SEQUENCE_LENGTH, N_JOINTS * 2) 
            for buf in delayed_buffers_list
        ])
        true_target_batch = np.array(true_targets_list)
        
        return delayed_seq_batch, true_target_batch
    
    except AttributeError as e:
        raise RuntimeError(
            f"Environment method not found: {e}\n"
            "Please ensure TeleoperationEnvWithDelay implements:\n"
            "  - get_delayed_target_buffer(length: int) -> np.ndarray\n"
            "  - get_true_current_target() -> np.ndarray"
        )


def pretrain_estimator(args: argparse.Namespace) -> None:
    """Main pre-training function for the State Estimator LSTM."""
    
    # Setup output directory and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Pretrain_LSTM_{args.config.name}_{timestamp}"
    output_dir = os.path.join(CHECKPOINT_DIR_LSTM, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    # Log essential configuration
    logger.info("="*80)
    logger.info("LSTM STATE ESTIMATOR PRE-TRAINING")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Delay Config: {args.config.name}")
    logger.info(f"  Trajectory: {args.trajectory_type.value}")
    logger.info(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"  Environments: {NUM_ENVIRONMENTS}")
    logger.info(f"  Sequence Length: {RNN_SEQUENCE_LENGTH}")
    logger.info(f"Hyperparameters:")
    logger.info(f"  Learning Rate: {ESTIMATOR_LEARNING_RATE}")
    logger.info(f"  Batch Size: {ESTIMATOR_BATCH_SIZE}")
    logger.info(f"  Buffer Size: {ESTIMATOR_BUFFER_SIZE}")
    logger.info(f"  Total Updates: {ESTIMATOR_TOTAL_UPDATES}")
    logger.info(f"  Warmup Steps: {ESTIMATOR_WARMUP_STEPS}")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    # Create vectorized environments
    def make_env():
        return TeleoperationEnvWithDelay(
            delay_config=args.config,
            trajectory_type=args.trajectory_type,
            randomize_trajectory=False,
            seed=args.seed
        )
    
    env = make_vec_env(
        make_env,
        n_envs=NUM_ENVIRONMENTS,
        seed=args.seed,
        vec_env_cls= DummyVecEnv
    )
    
    # Initialize State Estimator and optimizer
    state_estimator = StateEstimator().to(device)
    optimizer = torch.optim.Adam(state_estimator.parameters(), lr=ESTIMATOR_LEARNING_RATE)
    
    # Initialize replay buffer
    replay_buffer = PretrainReplayBuffer(ESTIMATOR_BUFFER_SIZE, device)
    
    obs = env.reset()
    
    for warmup_step in range(ESTIMATOR_WARMUP_STEPS):
        delayed_seq_batch, true_target_batch = collect_data_from_envs(env, NUM_ENVIRONMENTS)
        
        for i in range(NUM_ENVIRONMENTS):
            replay_buffer.add(delayed_seq_batch[i], true_target_batch[i])
        
        random_actions = np.array([env.action_space.sample() for _ in range(NUM_ENVIRONMENTS)])
        obs, rewards, dones, infos = env.step(random_actions)
    
    logger.info(f"Buffer filled: {len(replay_buffer)} samples")

    # Training
    state_estimator.train()
    
    best_loss = float('inf')
    loss_history = deque(maxlen=100)
    
    for update in range(ESTIMATOR_TOTAL_UPDATES):
        
        # Collect and add data
        delayed_seq_batch, true_target_batch = collect_data_from_envs(env, NUM_ENVIRONMENTS)
        
        for i in range(NUM_ENVIRONMENTS):
            replay_buffer.add(delayed_seq_batch[i], true_target_batch[i])
        
        random_actions = np.array([env.action_space.sample() for _ in range(NUM_ENVIRONMENTS)])
        obs, rewards, dones, infos = env.step(random_actions)
        
        # Training step
        batch = replay_buffer.sample(ESTIMATOR_BATCH_SIZE)
        predicted_targets, _ = state_estimator(batch['delayed_sequences'])
        loss = F.mse_loss(predicted_targets, batch['true_targets'])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state_estimator.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Logging
        if update % 1000 == 0:
            avg_loss = np.mean(loss_history) if len(loss_history) > 0 else loss.item()
            
            tb_writer.add_scalar('loss/mse', loss.item(), update)
            tb_writer.add_scalar('loss/avg_100', avg_loss, update)
            
            logger.info(f"Update {update}/{ESTIMATOR_TOTAL_UPDATES}: Loss={loss.item():.6f}, Avg={avg_loss:.6f}")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'state_estimator_state_dict': state_estimator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'update': update,
            }, os.path.join(output_dir, "estimator_best.pth"))
    
    # Save final model
    final_avg_loss = np.mean(loss_history) if len(loss_history) > 0 else loss.item()
    
    torch.save({
        'state_estimator_state_dict': state_estimator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': loss.item(),
        'final_avg_loss': final_avg_loss,
        'best_loss': best_loss,
        'config': {
            'delay_config': args.config.name,
            'trajectory_type': args.trajectory_type.value,
            'learning_rate': ESTIMATOR_LEARNING_RATE,
            'batch_size': ESTIMATOR_BATCH_SIZE,
            'total_updates': ESTIMATOR_TOTAL_UPDATES,
        }
    }, os.path.join(output_dir, "estimator_final.pth"))
    
    logger.info("="*80)
    logger.info("Training Complete")
    logger.info(f"Best Loss: {best_loss:.6f}")
    logger.info(f"Final Loss: {loss.item():.6f}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)
    
    env.close()
    tb_writer.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-train the State Estimator LSTM for teleoperation with delays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="2", 
        choices=['1', '2', '3', '4'], 
        help="Delay configuration preset."
    )
    parser.add_argument(
        "--trajectory-type", 
        type=str, 
        default="figure_8", 
        choices=[t.value for t in TrajectoryType], 
        help="Reference trajectory type."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed."
    )
    
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
    args = parse_arguments()
    
    try:
        pretrain_estimator(args)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise