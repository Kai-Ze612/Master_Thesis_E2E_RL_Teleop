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
        output_dim: 14D (predicted q + qd)
        """
        super().__init__()
        
        self.output_dim = cfg.ESTIMATOR_OUTPUT_DIM  # q and qd
        
        self.lstm = nn.LSTM(
            input_size=cfg.ESTIMATOR_STATE_DIM,
            hidden_size=cfg.RNN_HIDDEN_DIM, 
            num_layers=cfg.RNN_NUM_LAYERS,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(cfg.RNN_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
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
    """
    Helper function to filter out unstable trajectories.
    """
    # Check for NaNs and extreme values
    if np.isnan(delayed_seq).any() or np.isnan(true_target).any(): 
        return False
    # Check for extreme joint velocities
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
        self.ptr = 0  
        self.size = 0
        self.seq_len = cfg.RNN_SEQUENCE_LENGTH
        self.state_dim = cfg.ESTIMATOR_STATE_DIM 
        self.delayed_sequences = np.zeros((buffer_size, self.seq_len, self.state_dim), dtype=np.float32)
        
        # We still store true_targets for reference/validation, though we won't train on them directly
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
    delayed_flat_list = env.env_method("get_delayed_target_buffer", cfg.RNN_SEQUENCE_LENGTH)
    true_target_list = env.env_method("get_true_current_target")
    
    raw_seqs = np.array([buf.reshape(cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM) for buf in delayed_flat_list])
    raw_targets = np.array(true_target_list)
    
    valid_seqs, valid_targets = [], []
    for i in range(num_envs):
        if is_trajectory_stable(raw_seqs[i], raw_targets[i]):
            valid_seqs.append(raw_seqs[i])
            valid_targets.append(raw_targets[i])
            
    if len(valid_seqs) == 0: return np.array([]), np.array([])
    return np.array(valid_seqs), np.array(valid_targets)

def get_max_prediction_steps(delay_config: ExperimentConfig) -> int:
    """
    Retrieve max prediction horizon dynamically from DelaySimulator settings.
    """
    
    params = DelaySimulator._DELAY_CONFIGS[delay_config]
    max_delay_ms = params.obs_delay_max
    
    max_steps = int((max_delay_ms / 1000.0) * cfg.DEFAULT_CONTROL_FREQ)
    
    return max_steps + 2  # Add small buffer

def evaluate_model(model, val_env, num_val_steps, delay_config, device='cpu'):
    """
    Validation Loop:
    """
    model.eval()
    total_loss = 0.0
    count = 0
    
    val_env.reset()
    warmup_steps = int(cfg.WARM_UP_DURATION * cfg.DEFAULT_CONTROL_FREQ) + 20
    for _ in range(warmup_steps): 
        val_env.step([np.zeros((val_env.num_envs, cfg.N_JOINTS))])
    
    dt_norm = (1.0 / cfg.DEFAULT_CONTROL_FREQ) / cfg.DELAY_INPUT_NORM_FACTOR

    prediction_steps = get_max_prediction_steps(delay_config)

    with torch.no_grad():
        for i in range(num_val_steps):
            if i % 2000 == 0: print(f"Validating... {i}/{num_val_steps}", end='\r')

            seqs, true_targets = collect_data_from_envs(val_env, val_env.num_envs)
            if len(seqs) == 0: 
                val_env.step([np.zeros((val_env.num_envs, cfg.N_JOINTS))])
                continue
            
            input_t = torch.tensor(seqs, dtype=torch.float32).to(device)
            target_t = torch.tensor(true_targets, dtype=torch.float32).to(device)
            
            # Use dynamic context length
            context_len = cfg.RNN_SEQUENCE_LENGTH - prediction_steps
            context_input = input_t[:, :context_len, :]
            
            # Warmup
            _, hidden = model.lstm(context_input)
            current_input = context_input[:, -1:, :]
            
            # Predict forward
            for _ in range(prediction_steps):
                delta, hidden = model.forward_step(current_input, hidden)
                last_state = current_input[:, :, :14]
                next_state = last_state + (delta.unsqueeze(1) * cfg.TARGET_DELTA_SCALE)
                
                curr_delay = current_input[:, :, 14]
                next_delay = curr_delay + dt_norm
                current_input = torch.cat([next_state, next_delay.unsqueeze(2)], dim=2)
            
            # Compare against the actual future step in the sequence (Self-Supervised Validation)
            ground_truth_idx = min(context_len + prediction_steps - 1, cfg.RNN_SEQUENCE_LENGTH - 1)
            ground_truth_future = input_t[:, ground_truth_idx, :14]
            
            loss = F.mse_loss(current_input[:, 0, :14], ground_truth_future)
            total_loss += loss.item()
            count += 1
            
            val_env.step([np.zeros((val_env.num_envs, cfg.N_JOINTS))])
            
    model.train()
    print(f"Validating... Done.                 ")
    return total_loss / max(count, 1)


def get_saved_config() -> dict:
    """
    FIX Issue 6: Create config dict to save with checkpoint.
    """
    return {
        'DELAY_INPUT_NORM_FACTOR': cfg.DELAY_INPUT_NORM_FACTOR,
        'TARGET_DELTA_SCALE': cfg.TARGET_DELTA_SCALE,
        'RNN_HIDDEN_DIM': cfg.RNN_HIDDEN_DIM,
        'RNN_NUM_LAYERS': cfg.RNN_NUM_LAYERS,
        'RNN_SEQUENCE_LENGTH': cfg.RNN_SEQUENCE_LENGTH,
        'DEFAULT_CONTROL_FREQ': cfg.DEFAULT_CONTROL_FREQ,
        'ESTIMATOR_STATE_DIM': cfg.ESTIMATOR_STATE_DIM,
        'N_JOINTS': cfg.N_JOINTS,
    }


def train(args: argparse.Namespace) -> None:
    """Main training loop for autoregressive LSTM state estimator."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Autoregressive_LSTM_{args.config.name}_{timestamp}"
    output_dir = os.path.join(cfg.CHECKPOINT_DIR_LSTM, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)    
    
    # Log config for reproducibility
    saved_config = get_saved_config()
    logger.info("Training Config:")
    for key, value in saved_config.items():
        logger.info(f"  {key}: {value}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
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
    val_env = SubprocVecEnv([make_env(i + 1000) for i in range(cfg.NUM_ENVIRONMENTS)])
    
    replay_buffer = ReplayBuffer(cfg.ESTIMATOR_BUFFER_SIZE, device)
    model = AutoregressiveStateEstimator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ESTIMATOR_LEARNING_RATE)
    
    logger.info("Collecting Initial Data...")
    train_env.reset()
    init_warmup = int(cfg.WARM_UP_DURATION * cfg.DEFAULT_CONTROL_FREQ) + 50
    for _ in range(init_warmup): train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))]) 
    
    collected = 0
    while collected < cfg.ESTIMATOR_BUFFER_SIZE // 5:
        seqs, targets = collect_data_from_envs(train_env, cfg.NUM_ENVIRONMENTS)
        if len(seqs) > 0:
            for i in range(len(seqs)): replay_buffer.add(seqs[i], targets[i])
            collected += len(seqs)
            train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])
        else:
            train_env.reset()
            for _ in range(50): train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])

    logger.info(f"Buffer filled. Starting Training Loop.")
    
    max_arg_steps = get_max_prediction_steps(args.config)
    
    best_val_loss = float('inf')
    patience_counter = 0
    dt_norm = (1.0 / cfg.DEFAULT_CONTROL_FREQ) / cfg.DELAY_INPUT_NORM_FACTOR
    
    model.train()
    
    for update in range(1, cfg.ESTIMATOR_TOTAL_UPDATES + 1):
        # Data Collection
        seqs, targets = collect_data_from_envs(train_env, cfg.NUM_ENVIRONMENTS)
        if len(seqs) > 0:
            for i in range(len(seqs)): replay_buffer.add(seqs[i], targets[i])
        train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])
        
        # Training Step
        batch = replay_buffer.sample(cfg.ESTIMATOR_BATCH_SIZE)
        input_seq = batch['delayed_sequences']
        
        packet_loss_steps = np.random.randint(1, max_arg_steps)
        
        # clear out the last k steps to simulate packet loss
        cutoff_idx = cfg.RNN_SEQUENCE_LENGTH - packet_loss_steps
        safe_history = input_seq[:, :cutoff_idx, :]
        target_state_future = input_seq[:, -1, :14]
        
        # Warmup on history
        _, hidden_state = model.lstm(safe_history)
        current_input = safe_history[:, -1:, :]
        
        # Autoregressive rollout
        for k in range(packet_loss_steps):
            pred_delta, hidden_state = model.forward_step(current_input, hidden_state)
            
            last_known_state = current_input[:, :, :14] 
            predicted_next_state = last_known_state + (pred_delta.unsqueeze(1) * cfg.TARGET_DELTA_SCALE)
            
            current_delay_norm = current_input[:, :, 14]
            next_delay_norm = current_delay_norm + dt_norm
            
            next_input = torch.cat([predicted_next_state, next_delay_norm.unsqueeze(2)], dim=2)
            current_input = next_input 
            
        final_predicted_state = current_input[:, 0, :14]
        
        # Compute Loss
        loss = F.l1_loss(final_predicted_state, target_state_future)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if update % 100 == 0:
            tb_writer.add_scalar('loss/train', loss.item(), update)
            print(f"Update {update} | Loss: {loss.item():.6f}")
            
        if update % cfg.ESTIMATOR_VAL_FREQ == 0:
            val_loss = evaluate_model(model, val_env, num_val_steps=2000, delay_config=args.config, device=device)
            tb_writer.add_scalar('loss/validation', val_loss, update)
            logger.info(f"Validating at {update}: Val Loss = {val_loss:.6f} (Best: {best_val_loss:.6f})")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_path = os.path.join(output_dir, "autoregressive_estimator.pth")
                
                # FIX Issue 6: Save config with checkpoint
                torch.save({
                    'state_estimator_state_dict': model.state_dict(),
                    'config': saved_config,
                    'best_val_loss': best_val_loss,
                    'update': update,
                }, save_path)
                logger.info(f"  -> New Best! Saved model with config.")
            else:
                patience_counter += 1
                if patience_counter >= cfg.ESTIMATOR_PATIENCE:
                    logger.info("Early Stopping Triggered.")
                    break
            
    # Save final model
    final_save_path = os.path.join(output_dir, "final_autoregressive_estimator.pth")
    torch.save({
        'state_estimator_state_dict': model.state_dict(),
        'config': saved_config,
        'final_val_loss': best_val_loss,
        'total_updates': update,
    }, final_save_path)
    
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
    train(parse_arguments())
