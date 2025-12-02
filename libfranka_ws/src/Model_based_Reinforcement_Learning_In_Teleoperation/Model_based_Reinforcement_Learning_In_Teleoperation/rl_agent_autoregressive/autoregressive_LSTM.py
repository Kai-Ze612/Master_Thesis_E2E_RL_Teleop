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

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

# Environment Creation
def make_env_factory(rank: int, config: ExperimentConfig, traj_type: TrajectoryType, seed: int, randomize: bool):
    """
    Factory function to create environments. 
    """
    def _init():
        return TeleoperationEnvWithDelay(
            delay_config=config,
            trajectory_type=traj_type,
            randomize_trajectory=randomize,
            seed=seed + rank,
            lstm_model_path=None
        )
    return _init


class ReplayBuffer:
    def __init__(self, buffer_size: int, device: torch.device):
        self.max_size = buffer_size
        self.device = device
        self.ptr = 0
        self.current_size = 0
        
        # Dimensions
        self.seq_length = cfg.RNN_SEQUENCE_LENGTH
        self.input_dim = cfg.ESTIMATOR_STATE_DIM  # 15D
        self.output_dim = cfg.ESTIMATOR_OUTPUT_DIM  # 14D (Single Step)

        # Pre-allocate memory
        self.input_sequences = np.zeros((buffer_size, self.seq_length, self.input_dim), dtype=np.float32)
        self.target_states = np.zeros((buffer_size, self.output_dim), dtype=np.float32)

    def add_batch(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        """
        Add a batch of collected experiences to the buffer.
        """
        batch_size = len(sequences)
        if batch_size == 0:
            return

        # Handle wrap-around indices
        indices = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        
        self.input_sequences[indices] = sequences
        self.target_states[indices] = targets
        
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.current_size = min(self.current_size + batch_size, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random batch for training.
        """
        indices = np.random.randint(0, self.current_size, size=batch_size)
        
        return (
            torch.tensor(self.input_sequences[indices], device=self.device),
            torch.tensor(self.target_states[indices], device=self.device)
        )

    @property
    def is_filled_min_threshold(self) -> bool:
        return self.current_size >= self.max_size // 5


class LSTMTrainer:
    """
    Main class managing the training lifecycle of the Autoregressive LSTM.
    """
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Setup Directory & Logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"LSTM_Step_{self.args.config.name}_{self.timestamp}"
        self.output_dir = os.path.join(cfg.CHECKPOINT_DIR_LSTM, self.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tensorboard"))
        
        self.logger.info(f"Initialized Training on {self.device}")
        self.logger.info(f"Config: {self.args.config.name} | Trajectory: {self.args.trajectory_type.value}")

        # 2. Setup Model & Optimization
        # Output dim is 14 because we predict 1 step: (q_next, qd_next)
        self.model = StateEstimator(output_dim=cfg.N_JOINTS * 2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.ESTIMATOR_LEARNING_RATE)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        
        # 3. Setup Buffer
        self.buffer = ReplayBuffer(cfg.ESTIMATOR_BUFFER_SIZE, self.device)
        
        # 4. Setup Environments
        self.train_env, self.val_env = self._setup_environments()
        
        # Training State
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
        """
        Initialize Training and Validation environments.
        Validation uses a different random seed to test generalization.
        """
        # Training Envs - Sequential Execution
        train_env = DummyVecEnv([
            make_env_factory(i, self.args.config, self.args.trajectory_type, self.args.seed, self.args.randomize_trajectory)
            for i in range(cfg.NUM_ENVIRONMENTS)
        ])
        
        # Validation Env (Offset seed by 10000 to ensure distinct trajectory parameters)
        val_env = DummyVecEnv([
            make_env_factory(10000, self.args.config, self.args.trajectory_type, self.args.seed, self.args.randomize_trajectory)
            for _ in range(1)
        ])
        
        return train_env, val_env

    def _collect_rollouts(self, env: DummyVecEnv) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interact with the environment to collect data.
        Returns: (Batch of Input Sequences, Batch of Target Next States)
        """
        # 1. Get Input: Delayed History Sequences (t-delay ... t)
        # Shape: (Num_Envs, Seq_Len, 15)
        delayed_flat_list = env.env_method("get_delayed_target_buffer", cfg.RNN_SEQUENCE_LENGTH)
        
        # 2. Get Target: Future Single Step (t+1)
        # Shape: (Num_Envs, 14)
        true_single_list = env.env_method("get_future_target_single")
        
        # 3. Reshape and Validate
        input_dim = cfg.ESTIMATOR_STATE_DIM
        
        raw_inputs = np.array([
            buf.reshape(cfg.RNN_SEQUENCE_LENGTH, input_dim) for buf in delayed_flat_list
        ])
        raw_targets = np.array(true_single_list)
        
        # Filter NaNs (Crucial for stability)
        valid_indices = []
        for i in range(len(raw_inputs)):
            if not np.isnan(raw_inputs[i]).any() and not np.isnan(raw_targets[i]).any():
                valid_indices.append(i)
                
        if not valid_indices:
            return np.array([]), np.array([])
            
        return raw_inputs[valid_indices], raw_targets[valid_indices]

    def _validate(self, num_samples: int = 500) -> float:
        """
        Run evaluation loop on the held-out validation environment.
        """
        self.model.eval()
        total_val_loss = 0.0
        samples_processed = 0
        
        # Reset to ensure clean state
        self.val_env.reset()
        
        with torch.no_grad():
            while samples_processed < num_samples:
                # Step env to progress simulation
                self.val_env.step([np.zeros((self.val_env.num_envs, cfg.N_JOINTS))])
                
                inputs, targets = self._collect_rollouts(self.val_env)
                if len(inputs) == 0:
                    continue
                
                input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
                target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)
                
                # Mixed Precision Inference
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    predicted_next_state, _ = self.model(input_tensor)
                    loss = F.mse_loss(predicted_next_state, target_tensor)
                
                total_val_loss += loss.item() * len(inputs)
                samples_processed += len(inputs)
                
        self.model.train()
        return total_val_loss / samples_processed if samples_processed > 0 else float('inf')

    def run(self):
        """
        Main execution loop.
        """
        self.logger.info(">>> Filling Replay Buffer...")
        self.train_env.reset()
        
        # Warmup Phase
        while not self.buffer.is_filled_min_threshold:
            inputs, targets = self._collect_rollouts(self.train_env)
            self.buffer.add_batch(inputs, targets)
            # Step environment to generate new data points
            self.train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])

        self.logger.info(">>> Buffer Filled. Starting Training...")
        self.model.train()
        
        # Training Loop
        for update in range(1, cfg.ESTIMATOR_TOTAL_UPDATES + 1):
            self.global_step = update
            
            # A. Data Collection Step
            inputs, targets = self._collect_rollouts(self.train_env)
            if len(inputs) > 0:
                self.buffer.add_batch(inputs, targets)
            
            # Step physics
            self.train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])
            
            # B. Gradient Descent Step
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
            
            # Logging
            self.tb_writer.add_scalar("Train/Loss", train_loss.item(), update)
            
            # C. Validation Step
            if update % cfg.ESTIMATOR_VAL_FREQ == 0:
                val_mse = self._validate()
                
                self.logger.info(f"Update {update} | Train MSE: {train_loss.item():.6f} | Val MSE: {val_mse:.6f}")
                self.tb_writer.add_scalar("Val/Loss", val_mse, update)
                
                # Checkpointing
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
    args = parser.parse_args()
    
    # Map Config Arguments
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2], '4': config_options[3]}
    args.config = CONFIG_MAP[args.config]
    
    # Map Trajectory Arguments
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    
    return args


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True) # <-- Not needed for DummyVecEnv
    args = parse_arguments()
    trainer = LSTMTrainer(args)
    trainer.run()