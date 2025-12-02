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
def make_env_factory(
    rank: int,
    config: ExperimentConfig,
    traj_type: TrajectoryType,
    seed: int,
    randomize: bool,
    render_mode: Optional[str] = None
    ):
    def _init():
        return TeleoperationEnvWithDelay(
            delay_config=config,
            trajectory_type=traj_type,
            randomize_trajectory=randomize,
            seed=seed + rank,
            render_mode=render_mode, 
            lstm_model_path=None # We inject the model manually during validation
        )
    return _init


class ReplayBuffer:
    """
    Modified to store multi-step targets for AR training.
    Target Shape: (Batch, Horizon, Output_Dim)
    """
    def __init__(self, buffer_size: int, device: torch.device):
        
        self.max_size = buffer_size
        self.device = device
        self.ptr = 0
        self.current_size = 0
        
        self.seq_length = cfg.RNN_SEQUENCE_LENGTH
        self.input_dim = cfg.ESTIMATOR_STATE_DIM
        self.output_dim = cfg.ESTIMATOR_OUTPUT_DIM
        self.ar_horizon = cfg.MAX_AR_STEPS

        self.input_sequences = np.zeros((self.max_size, self.seq_length, self.input_dim), dtype=np.float32)
        self.target_sequences = np.zeros((self.max_size, self.ar_horizon, self.output_dim), dtype=np.float32)

    def add_batch(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        batch_size = len(sequences)
        if batch_size == 0: return

        indices = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        self.input_sequences[indices] = sequences
        self.target_sequences[indices] = targets
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.current_size = min(self.current_size + batch_size, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.current_size, size=batch_size)
        return (
            torch.tensor(self.input_sequences[indices], device=self.device),
            torch.tensor(self.target_sequences[indices], device=self.device)
        )

    @property
    def is_filled_min_threshold(self) -> bool:
        return self.current_size >= self.max_size // 5


class LSTMTrainer:
    def __init__(self, args: argparse.Namespace):
        
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"LSTM_AR_{self.args.config.name}_{self.timestamp}"
        self.output_dir = os.path.join(cfg.CHECKPOINT_DIR_LSTM, self.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tensorboard"))
        
        self.model = StateEstimator(output_dim=int(cfg.N_JOINTS * 2)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.ESTIMATOR_LEARNING_RATE)
        
        self.buffer = ReplayBuffer(int(cfg.ESTIMATOR_BUFFER_SIZE), self.device)
        self.train_env, self.val_env = self._setup_environments()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Scheduled Sampling parameters
        self.ss_prob = 1.0 # Probability of using Ground Truth (Teacher Forcing). Decays to 0.
        self.ss_decay_rate = 0.9995

    def _setup_logging(self) -> logging.Logger:
        log_file = os.path.join(self.output_dir, "training.log")
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
        return logging.getLogger(__name__)

    def _setup_environments(self) -> Tuple[DummyVecEnv, DummyVecEnv]:
        render_mode = "human" if self.args.render else None
        
        # Training Environment (Vectorized)
        train_env = DummyVecEnv([
            make_env_factory(i, self.args.config, self.args.trajectory_type, self.args.seed, self.args.randomize_trajectory, render_mode)
            for i in range(cfg.NUM_ENVIRONMENTS)
        ])
        
        val_env = DummyVecEnv([
            make_env_factory(0, self.args.config, self.args.trajectory_type, 10000, self.args.randomize_trajectory, None)
            for _ in range(1)
        ])
        return train_env, val_env

    def _collect_rollouts(self, env: DummyVecEnv) -> Tuple[np.ndarray, np.ndarray]:
        
        # Explicit int casting to prevent TypeError
        delayed_flat_list = env.env_method("get_delayed_target_buffer", int(cfg.RNN_SEQUENCE_LENGTH))
        ar_targets_flat = env.env_method("get_future_target_sequence", int(cfg.MAX_AR_STEPS))
        
        input_dim = int(cfg.ESTIMATOR_STATE_DIM)
        output_dim = int(cfg.N_JOINTS * 2)
        
        raw_inputs = np.array([buf.reshape(int(cfg.RNN_SEQUENCE_LENGTH), input_dim) for buf in delayed_flat_list])
        raw_targets = np.array([buf.reshape(int(cfg.MAX_AR_STEPS), output_dim) for buf in ar_targets_flat])
        
        valid_indices = []
        for i in range(len(raw_inputs)):
            if not np.isnan(raw_inputs[i]).any() and not np.isnan(raw_targets[i]).any():
                valid_indices.append(i)
                
        if not valid_indices: return np.array([]), np.array([])
        return raw_inputs[valid_indices], raw_targets[valid_indices]

    def _autoregressive_loss(self, batch_inputs, batch_targets):
        """
        Calculates loss over the AR horizon with Scheduled Sampling.
        Logic matches deployment: decrements delay and reconstructs 15D input.
        """
        loss = 0
        
        # 1. Initial State (Process History)
        _, hidden = self.model.lstm(batch_inputs)
        
        curr_input = batch_inputs[:, -1:, :]
        curr_delay = curr_input[:, :, -1:] 
        dt_norm_step = (1.0 / cfg.DEFAULT_CONTROL_FREQ) / cfg.DELAY_INPUT_NORM_FACTOR
        
        horizon = batch_targets.shape[1]
        
        for t in range(horizon):
            pred_state, hidden = self.model.forward_step(curr_input, hidden) 
            
            gt_state = batch_targets[:, t:t+1, :] 
            
            # Loss is always calculated against Ground Truth
            loss += F.mse_loss(pred_state, gt_state)
            
            # Scheduled Sampling: Choose input for NEXT step
            use_ground_truth = (np.random.random() < self.ss_prob)
            
            if use_ground_truth:
                next_state = gt_state
            else:
                next_state = pred_state
            
            curr_delay = torch.clamp(curr_delay - dt_norm_step, min=0.0)
            curr_input = torch.cat([next_state, curr_delay], dim=2)

        return loss / horizon

    def _validate_full_trajectory(self, duration_sec: float = 60.0) -> float:
        """
        Rigorous Validation:
        Injects the current model into the Validation Environment and runs a 
        full 60-second episode. This forces the Env to use the internal 
        AR Prediction Loop (Pure Autoregression), testing true deployment stability.
        """
        self.model.eval()
        
        # 1. Inject Model into Env
        # We access the unwrapped env to set the 'lstm' attribute
        val_env_instance = self.val_env.envs[0].unwrapped
        
        # Save previous state just in case (though usually None)
        prev_lstm = val_env_instance.lstm
        val_env_instance.lstm = self.model 
        
        # 2. Run Episode
        steps_to_run = int(duration_sec * cfg.DEFAULT_CONTROL_FREQ)
        total_pred_error = 0.0
        
        # Reset with fixed seed implied by env creation (10000)
        self.val_env.reset()
        
        with torch.no_grad():
            for _ in range(steps_to_run):
                # Step with zero action (we only care about State Estimation here)
                _, _, done, infos = self.val_env.step([np.zeros(cfg.N_JOINTS)])
                
                # The Env calculates prediction error internally in _get_info
                # "prediction_error": distance between AR_Prediction and GT
                error = infos[0].get("prediction_error", 0.0)
                total_pred_error += error
                
                if done[0]:
                    self.val_env.reset()
                    
        # 3. Cleanup
        val_env_instance.lstm = prev_lstm
        self.model.train()
        
        avg_error = total_pred_error / steps_to_run
        return avg_error

    def run(self):
        self.logger.info(">>> Filling Replay Buffer...")
        self.train_env.reset()
        
        while not self.buffer.is_filled_min_threshold:
            inputs, targets = self._collect_rollouts(self.train_env)
            self.buffer.add_batch(inputs, targets)
            self.train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])

        self.logger.info(">>> Buffer Filled. Starting AR Training...")
        self.model.train()
        
        for update in range(1, cfg.ESTIMATOR_TOTAL_UPDATES + 1):
            self.global_step = update
            
            # Data Collection
            inputs, targets = self._collect_rollouts(self.train_env)
            if len(inputs) > 0: self.buffer.add_batch(inputs, targets)
            self.train_env.step([np.zeros((cfg.NUM_ENVIRONMENTS, cfg.N_JOINTS))])
            
            # Training Step
            batch_inputs, batch_targets = self.buffer.sample(cfg.ESTIMATOR_BATCH_SIZE)
            
            self.optimizer.zero_grad()
            train_loss = self._autoregressive_loss(batch_inputs, batch_targets)
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Decay Scheduled Sampling
            self.ss_prob = max(0.0, self.ss_prob * self.ss_decay_rate)
            
            self.tb_writer.add_scalar("Train/Loss", train_loss.item(), update)
            self.tb_writer.add_scalar("Train/SS_Prob", self.ss_prob, update)
            
            # Validation Step
            if update % cfg.ESTIMATOR_VAL_FREQ == 0:
                # Run the rigorous 60s test
                val_error = self._validate_full_trajectory(duration_sec=60.0)
                
                self.logger.info(f"Update {update} | Loss: {train_loss.item():.6f} | Val Error (60s avg): {val_error:.6f} | SS: {self.ss_prob:.4f}")
                self.tb_writer.add_scalar("Val/Prediction_Error_60s", val_error, update)
                
                # Save Best Model
                if val_error < self.best_val_loss:
                    self.best_val_loss = val_error
                    self.patience_counter = 0
                    self._save_checkpoint("best_model.pth")
                    self.logger.info(f"  [>] New Best Model Saved! ({val_error:.6f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= cfg.ESTIMATOR_PATIENCE:
                        self.logger.info("Early stopping triggered.")
                        break
                
                # Always save latest
                self._save_checkpoint("latest_model.pth")

        self._save_checkpoint("final_model.pth")
        self._close()

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), path)

    def _close(self):
        self.train_env.close()
        self.val_env.close()
        self.tb_writer.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Step-Based Autoregressive LSTM for State Estimation")
    parser.add_argument("--config", type=str, default="4", help="Delay Configuration ID")
    parser.add_argument("--trajectory-type", type=str, default="figure_8", help="Trajectory Type")
    parser.add_argument("--seed", type=int, default=50, help="Random Seed")
    parser.add_argument("--randomize-trajectory", action="store_true", help="Randomize trajectory parameters")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo rendering (Debug only, slows training)")
    
    args = parser.parse_args()
    
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {'1': config_options[0], '2': config_options[1], '3': config_options[2], '4': config_options[3]}
    args.config = CONFIG_MAP[args.config]
    
    args.trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())
    
    return args

if __name__ == "__main__":
    args = parse_arguments() 
    trainer = LSTMTrainer(args)
    trainer.run()
