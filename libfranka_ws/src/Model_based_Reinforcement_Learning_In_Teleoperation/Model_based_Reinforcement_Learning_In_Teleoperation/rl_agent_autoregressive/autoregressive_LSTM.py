"""
Autoregressive LSTM state estimator training script.

Features:
1. Early fusion LSTM:
    - Inputs: Sequence of delayed observations + normalized delay value (15D: 7D q, 7D qd, 1D delay)
    - Delay is encoded because the LSTM needs to know how far back in time the last observation was.
2. Autoregressive LSTM:
    - One-step prediction: Predicts the next robot state (q, qd) from a sequence of delayed observations.
    - Continuous prediction: If a new observation is unavailable due to delay, the model uses its last predicted state as the new input. 
    - This allows it to continuously forecast the robot's state, bridging the gap until a real observation arrives.
    
Input: sequence of 15D delayed sequence
Ouput: 14D true local robot state at real time
Loss: MSE Loss
"""


import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Tuple, Dict, Optional, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv 

# Project Imports
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg


# --- Helper Function for Environment Creation ---
def make_env_factory(rank: int, config: ExperimentConfig, traj_type: TrajectoryType, seed: int, randomize: bool, render_mode: Optional[str] = None):
    """
    Factory function to create environments. 
    """
    def _init():
        return TeleoperationEnvWithDelay(
            delay_config=config,
            trajectory_type=traj_type,
            randomize_trajectory=randomize,
            seed=seed + rank,
            render_mode=render_mode, # Pass render mode here
            lstm_model_path=None
        )
    return _init


class SequenceReplayBuffer:
    def __init__(self, buffer_size: int, device: torch.device):
        self.max_size = buffer_size
        self.device = device
        self.ptr = 0
        self.current_size = 0
        
        self.seq_length = cfg.RNN_SEQUENCE_LENGTH
        self.input_dim = cfg.ESTIMATOR_STATE_DIM 
        self.output_dim = cfg.N_JOINTS * 2 

        self.input_sequences = np.zeros((buffer_size, self.seq_length, self.input_dim), dtype=np.float32)
        self.target_states = np.zeros((buffer_size, self.output_dim), dtype=np.float32)

    def add_batch(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        batch_size = len(sequences)
        if batch_size == 0: return

        indices = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        self.input_sequences[indices] = sequences
        self.target_states[indices] = targets
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.current_size = min(self.current_size + batch_size, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.current_size, size=batch_size)
        return (
            torch.tensor(self.input_sequences[indices], device=self.device),
            torch.tensor(self.target_states[indices], device=self.device)
        )

    @property
    def is_filled_min_threshold(self) -> bool:
        return self.current_size >= self.max_size // 5


class LSTMTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"LSTM_Step_{self.args.config.name}_{self.timestamp}"
        self.output_dir = os.path.join(cfg.CHECKPOINT_DIR_LSTM, self.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tensorboard"))
        
        self.logger.info(f"Initialized Training on {self.device}")
        self.logger.info(f"Config: {self.args.config.name} | Trajectory: {self.args.trajectory_type.value}")
        if self.args.render:
            self.logger.info("Visual Rendering ENABLED (expect slower training)")

        self.model = StateEstimator(output_dim=cfg.N_JOINTS * 2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.ESTIMATOR_LEARNING_RATE)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        
        self.buffer = SequenceReplayBuffer(cfg.ESTIMATOR_BUFFER_SIZE, self.device)
        self.train_env, self.val_env = self._setup_environments()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _setup_logging(self) -> logging.Logger:
        log_file = os.path.join(self.output_dir, "training.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)

    def _setup_environments(self) -> Tuple[DummyVecEnv, DummyVecEnv]:
        # Determine render mode based on flag
        render_mode = "human" if self.args.render else None
        
        # Training Envs
        train_env = DummyVecEnv([
            make_env_factory(i, self.args.config, self.args.trajectory_type, self.args.seed, self.args.randomize_trajectory, render_mode)
            for i in range(cfg.NUM_ENVIRONMENTS)
        ])
        
        # Validation Env (No rendering for validation generally, unless debugging)
        val_env = DummyVecEnv([
            make_env_factory(10000, self.args.config, self.args.trajectory_type, self.args.seed, self.args.randomize_trajectory, None)
            for _ in range(1)
        ])
        
        return train_env, val_env

    def _collect_rollouts(self, env: DummyVecEnv) -> Tuple[np.ndarray, np.ndarray]:
        delayed_flat_list = env.env_method("get_delayed_target_buffer", cfg.RNN_SEQUENCE_LENGTH)
        true_single_list = env.env_method("get_future_target_single")
        
        input_dim = cfg.ESTIMATOR_STATE_DIM
        raw_inputs = np.array([buf.reshape(cfg.RNN_SEQUENCE_LENGTH, input_dim) for buf in delayed_flat_list])
        raw_targets = np.array(true_single_list)
        
        valid_indices = []
        for i in range(len(raw_inputs)):
            if not np.isnan(raw_inputs[i]).any() and not np.isnan(raw_targets[i]).any():
                valid_indices.append(i)
                
        if not valid_indices: return np.array([]), np.array([])
        return raw_inputs[valid_indices], raw_targets[valid_indices]

    def _validate(self, num_samples: int = 500) -> float:
        self.model.eval()
        total_val_loss = 0.0
        samples_processed = 0
        self.val_env.reset()
        
        with torch.no_grad():
            while samples_processed < num_samples:
                self.val_env.step([np.zeros((self.val_env.num_envs, cfg.N_JOINTS))])
                inputs, targets = self._collect_rollouts(self.val_env)
                if len(inputs) == 0: continue
                
                input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
                target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)
                
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    predicted_next_state, _ = self.model(input_tensor)
                    loss = F.mse_loss(predicted_next_state, target_tensor)
                
                total_val_loss += loss.item() * len(inputs)
                samples_processed += len(inputs)
                
        self.model.train()
        return total_val_loss / samples_processed if samples_processed > 0 else float('inf')

    def run(self):
        self.logger.info(">>> Filling Replay Buffer...")
        self.train_env.reset()
        
        while not self.buffer.is_filled_min_threshold:
            inputs, targets = self._collect_rollouts(self.train_env)
            self.buffer.add_batch(inputs, targets)
            self.train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])

        self.logger.info(">>> Buffer Filled. Starting Training...")
        self.model.train()
        
        for update in range(1, cfg.ESTIMATOR_TOTAL_UPDATES + 1):
            self.global_step = update
            
            inputs, targets = self._collect_rollouts(self.train_env)
            if len(inputs) > 0: self.buffer.add_batch(inputs, targets)
            self.train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])
            
            batch_inputs, batch_targets = self.buffer.sample(cfg.ESTIMATOR_BATCH_SIZE)
            
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                predictions, _ = self.model(batch_inputs)
                train_loss = F.mse_loss(predictions, batch_targets)
            
            self.scaler.scale(train_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.tb_writer.add_scalar("Train/Loss", train_loss.item(), update)
            
            if update % cfg.ESTIMATOR_VAL_FREQ == 0:
                val_mse = self._validate()
                self.logger.info(f"Update {update} | Train MSE: {train_loss.item():.6f} | Val MSE: {val_mse:.6f}")
                self.tb_writer.add_scalar("Val/Loss", val_mse, update)
                
                if val_mse < self.best_val_loss:
                    self.best_val_loss = val_mse
                    self.patience_counter = 0
                    self._save_checkpoint("best_model.pth")
                    self.logger.info(f"  [>] Best Model Saved! (Val MSE: {self.best_val_loss:.6f})")
                else:
                    self.patience_counter += 1
                    self.logger.info(f"  [!] No improvement. Patience: {self.patience_counter}/{cfg.ESTIMATOR_PATIENCE}")
                    if self.patience_counter >= cfg.ESTIMATOR_PATIENCE:
                        self.logger.info(">>> Early Stopping Triggered.")
                        break

        self._save_checkpoint("final_model.pth")
        self._close()

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), path)

    def _close(self):
        self.train_env.close()
        self.val_env.close()
        self.tb_writer.close()
        self.logger.info("Training Finished.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Step-Based Autoregressive LSTM for State Estimation")
    parser.add_argument("--config", type=str, default="4", help="Delay Configuration ID")
    parser.add_argument("--trajectory-type", type=str, default="figure_8", help="Trajectory Type")
    parser.add_argument("--seed", type=int, default=50, help="Random Seed")
    parser.add_argument("--randomize-trajectory", action="store_true", help="Randomize trajectory parameters")
    
    # NEW ARGUMENT: --render
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo rendering (Debug only, slows training)")
    
    args = parser.parse_args()
    
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2], '4': config_options[3]}
    args.config = CONFIG_MAP[args.config]
    
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    
    return args


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True) # Not needed for DummyVecEnv
    args = parse_arguments()
    trainer = LSTMTrainer(args)
    trainer.run()