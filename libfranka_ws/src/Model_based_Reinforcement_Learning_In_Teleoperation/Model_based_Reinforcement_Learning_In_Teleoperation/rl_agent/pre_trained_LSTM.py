"""
Simplified LSTM State Estimator Pre-Training Pipeline

This script implements a straightforward, debuggable approach:
1. Initialize LocalRobotSimulator directly (no vec environments)
2. Collect trajectory data (q_t, qd_t) from the simulator
3. Apply manual delays using DelaySimulator
4. Build delayed observation sequences
5. Pre-train LSTM via supervised learning to minimize MSE between
   LSTM-predicted state and true current state

Key advantages over the vectorized environment approach:
- Direct control over data collection process
- Easier debugging and visualization
- Clear separation of concerns (trajectory generation → delay simulation → training)
- Deterministic replay buffer filling
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import logging
from datetime import datetime
from collections import deque
import warnings
from typing import Dict, Tuple, Optional

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================================================
# Configuration and Constants
# ============================================================================

# These should match your actual config
N_JOINTS = 7
RNN_SEQUENCE_LENGTH = 64
RNN_HIDDEN_DIM = 512
RNN_NUM_LAYERS = 4
ESTIMATOR_LEARNING_RATE = 1e-4
ESTIMATOR_BATCH_SIZE = 512
ESTIMATOR_BUFFER_SIZE = 200000
ESTIMATOR_WARMUP_STEPS = 100000      # 20x larger sample pool!
ESTIMATOR_TOTAL_UPDATES = 200000     
ESTIMATOR_VAL_STEPS = 5000
ESTIMATOR_VAL_FREQ = 1000
ESTIMATOR_PATIENCE = 10
ESTIMATOR_LR_PATIENCE = 5
DEFAULT_CONTROL_FREQ = 500
LOG_STD_MIN = -20
LOG_STD_MAX = 2
SAC_ACTIVATION = 'relu'


# ============================================================================
# LSTM State Estimator Network
# ============================================================================

class StateEstimator(nn.Module):
    """
    LSTM-based state estimator.
    Input: sequence of delayed observations (batch, seq_len, 14)
    Output: predicted current state (batch, 14)
    """
    
    def __init__(
        self,
        input_dim: int = N_JOINTS * 2,
        hidden_dim: int = RNN_HIDDEN_DIM,
        num_layers: int = RNN_NUM_LAYERS,
        output_dim: int = N_JOINTS * 2,
    ):
        super().__init__()
        
        self.rnn_hidden_dim = hidden_dim
        self.rnn_num_layers = num_layers
        self.activation_fn = self._get_activation(SAC_ACTIVATION)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            self.activation_fn(),
            nn.Linear(128, output_dim)
        )
        self._initialize_weights()

    def _get_activation(self, activation_name: str):
        if activation_name == "relu": 
            return nn.ReLU
        elif activation_name == "tanh": 
            return nn.Tanh
        elif activation_name == "elu": 
            return nn.ELU
        else: 
            raise ValueError(f"Unsupported activation: {activation_name}")
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def init_hidden_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)
        c_0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)
        return (h_0, c_0)

    def forward(self, 
                delayed_sequence: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        lstm_output, new_hidden_state = self.lstm(delayed_sequence, hidden_state)
        last_lstm_output = lstm_output[:, -1, :]
        predicted_state = self.prediction_head(last_lstm_output)
        return predicted_state, new_hidden_state


# ============================================================================
# Replay Buffer for Training Data
# ============================================================================

class PretrainReplayBuffer:
    """
    Replay buffer for storing (delayed_sequence, true_target) pairs.
    
    Structure:
        - delayed_sequences: (buffer_size, seq_len, state_dim) = (B, 256, 14)
        - true_targets: (buffer_size, state_dim) = (B, 14)
    """
    
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
        """
        Add a single training example to the buffer.
        
        Args:
            delayed_seq: shape (seq_len, state_dim) - sequence of delayed observations
            true_target: shape (state_dim,) - current ground truth state [q, qd]
        """
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


# ============================================================================
# Data Collection from LocalRobotSimulator
# ============================================================================

class SimpleDataCollector:
    """
    Direct data collection from LocalRobotSimulator without vectorization.
    
    Pipeline:
        1. Run LocalRobotSimulator for N steps
        2. Collect full trajectory: [q_0, q_1, ..., q_N] and [qd_0, qd_1, ..., qd_N]
        3. For each timestep t:
           - Apply delay offset to get observation_index = t - delay_steps
           - Extract delayed window: [q[t-d-seq_len], ..., q[t-d]]
           - Store (delayed_window, [q[t], qd[t]]) as training pair
    """
    
    def __init__(
        self,
        control_freq: int = DEFAULT_CONTROL_FREQ,
        seed: Optional[int] = None
    ):
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.rng = np.random.RandomState(seed)
        
    def collect_trajectory(
        self,
        simulator,
        num_steps: int,
        delay_config: 'DelaySimulator'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect a complete trajectory from the simulator.
        
        Returns:
            q_trajectory: shape (num_steps, n_joints)
            qd_trajectory: shape (num_steps, n_joints)
        """
        q_trajectory = []
        qd_trajectory = []
        
        # Reset simulator
        q, info = simulator.reset(seed=np.random.randint(0, 10000))
        q_trajectory.append(q.copy())
        qd_trajectory.append(np.zeros(N_JOINTS))
        
        # Run trajectory
        for step in range(num_steps):
            q, qd, _, _, _, info = simulator.step()
            q_trajectory.append(q.copy())
            qd_trajectory.append(qd.copy())
        
        return np.array(q_trajectory), np.array(qd_trajectory)
    
    def build_delayed_sequences(
        self,
        q_trajectory: np.ndarray,
        qd_trajectory: np.ndarray,
        delay_config: 'DelaySimulator',
        buffer_length: int = RNN_SEQUENCE_LENGTH
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build training pairs: (delayed_observation_sequence, true_current_state)
        
        For each timestep t in trajectory:
            1. Sample observation delay: delay_steps ~ U[delay_min, delay_max]
            2. Observation index: obs_idx = t - delay_steps
            3. Delayed sequence window: [obs_idx - buffer_len, ..., obs_idx]
            4. Current true state: [q[t], qd[t]]
        
        Returns:
            delayed_seqs: shape (num_valid_timesteps, buffer_length, state_dim)
            true_targets: shape (num_valid_timesteps, state_dim)
        """
        num_steps = len(q_trajectory)
        delayed_seqs_list = []
        true_targets_list = []
        
        for t in range(buffer_length, num_steps):  # Start from buffer_length to allow for history
            # Sample random delay for this timestep
            delay_steps = self.rng.randint(
                delay_config._obs_delay_min_steps,
                delay_config._obs_delay_max_steps + 1
            )
            
            # Observation index with delay
            obs_idx = t - delay_steps
            if obs_idx < 0:
                obs_idx = 0
            
            # Extract delayed observation sequence
            seq_start = max(0, obs_idx - buffer_length + 1)
            seq_end = obs_idx + 1
            
            delayed_seq = []
            for idx in range(seq_start, seq_end):
                # Pad if necessary
                if idx < 0:
                    state = np.concatenate([np.zeros(N_JOINTS), np.zeros(N_JOINTS)])
                else:
                    state = np.concatenate([q_trajectory[idx], qd_trajectory[idx]])
                delayed_seq.append(state)
            
            # Pad sequence to fixed length if needed
            while len(delayed_seq) < buffer_length:
                delayed_seq.insert(0, np.concatenate([np.zeros(N_JOINTS), np.zeros(N_JOINTS)]))
            
            delayed_seq_array = np.array(delayed_seq)  # shape: (buffer_length, state_dim)
            
            # True current state at timestep t
            true_target = np.concatenate([q_trajectory[t], qd_trajectory[t]])
            
            delayed_seqs_list.append(delayed_seq_array)
            true_targets_list.append(true_target)
        
        return np.array(delayed_seqs_list), np.array(true_targets_list)


# ============================================================================
# Training Pipeline
# ============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to file and console."""
    log_file = os.path.join(output_dir, "lstm_pretrain.log")
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


def evaluate_model(
    model: StateEstimator,
    val_buffer: PretrainReplayBuffer,
    batch_size: int,
    num_batches: int = 50,
    device: torch.device = None
) -> float:
    """Evaluate model on validation buffer."""
    model.eval()
    total_loss = 0.0
    
    if len(val_buffer) < batch_size:
        return float('inf')
    
    with torch.no_grad():
        for _ in range(num_batches):
            batch = val_buffer.sample(batch_size)
            predicted_targets, _ = model(batch['delayed_sequences'])
            loss = F.mse_loss(predicted_targets, batch['true_targets'])
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


def pretrain_estimator_simple(
    output_dir: str,
    delay_config,
    trajectory_type,
    device: torch.device,
    seed: int = 42
) -> None:
    """
    Simplified pre-training loop without vectorized environments.
    """
    
    # Setup logging
    logger = setup_logging(output_dir)
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    logger.info("="*80)
    logger.info("LSTM STATE ESTIMATOR PRE-TRAINING (SIMPLIFIED)")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Delay Config: {delay_config.config_name}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Sequence Length: {RNN_SEQUENCE_LENGTH}")
    logger.info(f"Hyperparameters:")
    logger.info(f"  Learning Rate: {ESTIMATOR_LEARNING_RATE}")
    logger.info(f"  Batch Size: {ESTIMATOR_BATCH_SIZE}")
    logger.info(f"  Buffer Size: {ESTIMATOR_BUFFER_SIZE}")
    logger.info(f"  Total Updates: {ESTIMATOR_TOTAL_UPDATES}")
    logger.info("="*80)
    
    # Initialize model
    state_estimator = StateEstimator(
        input_dim=N_JOINTS * 2,
        hidden_dim=RNN_HIDDEN_DIM,
        num_layers=RNN_NUM_LAYERS,
        output_dim=N_JOINTS * 2,
    ).to(device)
    
    optimizer = optim.Adam(state_estimator.parameters(), lr=ESTIMATOR_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=ESTIMATOR_LR_PATIENCE,
    )
   
    # Initialize buffers
    replay_buffer = PretrainReplayBuffer(ESTIMATOR_BUFFER_SIZE, device)
    val_buffer = PretrainReplayBuffer(ESTIMATOR_VAL_STEPS, device)
    
    # Import components
    try:
        from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import LocalRobotSimulator, TrajectoryType
        from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
    except ImportError:
        logger.error("Failed to import required modules. Please ensure paths are correct.")
        raise
    
    # Initialize data collector
    collector = SimpleDataCollector(control_freq=DEFAULT_CONTROL_FREQ, seed=seed)
    
    # Initialize simulators
    logger.info("Initializing simulators...")
    train_sim = LocalRobotSimulator(trajectory_type=trajectory_type, randomize_params=True)
    val_sim = LocalRobotSimulator(trajectory_type=trajectory_type, randomize_params=False)
    
    # ========================================================================
    # PHASE 1: Fill Training Buffer
    # ========================================================================
    logger.info(f"Filling training buffer (target: {ESTIMATOR_WARMUP_STEPS} samples)...")
    
    samples_collected = 0
    episode = 0
    
    while samples_collected < ESTIMATOR_WARMUP_STEPS:
        # Collect trajectory
        trajectory_length = min(1000, ESTIMATOR_WARMUP_STEPS - samples_collected + RNN_SEQUENCE_LENGTH)
        q_traj, qd_traj = collector.collect_trajectory(
            train_sim,
            trajectory_length,
            delay_config
        )
        
        # Build training pairs
        delayed_seqs, true_targets = collector.build_delayed_sequences(
            q_traj, qd_traj, delay_config, buffer_length=RNN_SEQUENCE_LENGTH
        )
        
        # Add to buffer
        for delayed_seq, true_target in zip(delayed_seqs, true_targets):
            replay_buffer.add(delayed_seq, true_target)
            samples_collected += 1
            
            if samples_collected >= ESTIMATOR_WARMUP_STEPS:
                break
        
        episode += 1
        if episode % 5 == 0:
            logger.info(f"  Episode {episode}: Collected {samples_collected}/{ESTIMATOR_WARMUP_STEPS} samples")
    
    logger.info(f"Training buffer filled: {len(replay_buffer)} samples")
    
    # ========================================================================
    # PHASE 2: Fill Validation Buffer
    # ========================================================================
    logger.info(f"Filling validation buffer (target: {ESTIMATOR_VAL_STEPS} samples)...")
    
    samples_collected = 0
    episode = 0
    
    while samples_collected < ESTIMATOR_VAL_STEPS:
        trajectory_length = min(1000, ESTIMATOR_VAL_STEPS - samples_collected + RNN_SEQUENCE_LENGTH)
        q_traj, qd_traj = collector.collect_trajectory(
            val_sim,
            trajectory_length,
            delay_config
        )
        
        delayed_seqs, true_targets = collector.build_delayed_sequences(
            q_traj, qd_traj, delay_config, buffer_length=RNN_SEQUENCE_LENGTH
        )
        
        for delayed_seq, true_target in zip(delayed_seqs, true_targets):
            val_buffer.add(delayed_seq, true_target)
            samples_collected += 1
            
            if samples_collected >= ESTIMATOR_VAL_STEPS:
                break
        
        episode += 1
    
    logger.info(f"Validation buffer filled: {len(val_buffer)} samples")
    
    # ========================================================================
    # PHASE 3: Training Loop
    # ========================================================================
    logger.info("Starting training loop...")
    state_estimator.train()
    
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history = deque(maxlen=100)
    
    for update in range(ESTIMATOR_TOTAL_UPDATES):
        # Collect more data (online)
        if update % 100 == 0:
            q_traj, qd_traj = collector.collect_trajectory(
                train_sim,
                500,
                delay_config
            )
            delayed_seqs, true_targets = collector.build_delayed_sequences(
                q_traj, qd_traj, delay_config, buffer_length=RNN_SEQUENCE_LENGTH
            )
            for delayed_seq, true_target in zip(delayed_seqs, true_targets):
                replay_buffer.add(delayed_seq, true_target)
        
        # Training step
        batch = replay_buffer.sample(ESTIMATOR_BATCH_SIZE)
        predicted_targets, _ = state_estimator(batch['delayed_sequences'])
        loss = F.mse_loss(predicted_targets, batch['true_targets'])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state_estimator.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Validation and logging
        if update % ESTIMATOR_VAL_FREQ == 0:
            avg_train_loss = np.mean(loss_history) if len(loss_history) > 0 else loss.item()
            val_loss = evaluate_model(state_estimator, val_buffer, ESTIMATOR_BATCH_SIZE, device=device)
            
            tb_writer.add_scalar('loss/train_avg_100', avg_train_loss, update)
            tb_writer.add_scalar('loss/validation_mse', val_loss, update)
            tb_writer.add_scalar('hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], update)
            
            logger.info(f"Update {update}/{ESTIMATOR_TOTAL_UPDATES}: "
                       f"TrainLoss={avg_train_loss:.6f}, ValLoss={val_loss:.6f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"  -> New best validation loss. Saving model.")
                torch.save({
                    'state_estimator_state_dict': state_estimator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'update': update,
                }, os.path.join(output_dir, "estimator_best.pth"))
            else:
                patience_counter += 1
            
            if patience_counter >= ESTIMATOR_PATIENCE:
                logger.info(f"Early stopping triggered after {ESTIMATOR_PATIENCE} checks without improvement.")
                break
    
    # ========================================================================
    # Final Evaluation
    # ========================================================================
    logger.info("="*80)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best Validation Loss: {best_val_loss:.6f}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*80)
    
    tb_writer.close()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified LSTM pre-training")
    parser.add_argument("--config", type=str, default="3", 
                       choices=['1', '2', '3', '4'],
                       help="Delay configuration: 1=LOW, 2=MEDIUM, 3=HIGH, 4=FULL_RANGE")
    parser.add_argument("--trajectory", type=str, default="figure_8",
                       choices=['figure_8', 'square', 'lissajous_complex'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./lstm_training_output_simplified")
    
    args = parser.parse_args()
    
    # Setup paths
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"LSTM_Simplified_{args.config}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Import delay config
        from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig, DelaySimulator
        from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
        
        # Map config
        config_map = {
            '1': ExperimentConfig.LOW_DELAY,
            '2': ExperimentConfig.MEDIUM_DELAY,
            '3': ExperimentConfig.HIGH_DELAY,
            '4': ExperimentConfig.FULL_RANGE_COVER,
        }
        delay_config = DelaySimulator(DEFAULT_CONTROL_FREQ, config_map[args.config], seed=args.seed)
        trajectory_type = TrajectoryType(args.trajectory)
        
        # Run training
        pretrain_estimator_simple(
            output_dir=output_dir,
            delay_config=delay_config,
            trajectory_type=trajectory_type,
            device=device,
            seed=args.seed
        )
        
    except Exception as e:
        print(f"Error: {e}")
        raise