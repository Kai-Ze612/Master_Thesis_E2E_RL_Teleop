# File: measure_inference_time.py
#
# Description:
#   Measures inference time for the State Estimator (LSTM) and
#   Actor (SAC) models. This script is pre-filled with parameters
#   from your robot_config.py.
#
#   Usage:
#   1. Ensure the model paths in '--- CONFIGURATION ---' are correct.
#   2. Run the script: python measure_inference_time.py
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import time
import os
from typing import Tuple, Optional, Union

# ==================================================================
# --- CONFIGURATION ---
# Values from your robot_config.py and provided paths
# ==================================================================

# Paths to your trained models
ESTIMATOR_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent/lstm_training_output/Pretrain_LSTM_HIGH_DELAY_20251114_120334/estimator_best.pth"
POLICY_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent/rl_training_output/ModelBasedSAC_HIGH_DELAY_figure_8_20251114_220018/best_policy.pth"

# Robot parameters from robot_config.py
N_JOINTS = 7
MAX_TORQUE_COMPENSATION = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float32)

# LSTM parameters from robot_config.py
RNN_HIDDEN_DIM = 512
RNN_NUM_LAYERS = 4

# SAC parameters from robot_config.py
SAC_MLP_HIDDEN_DIMS = [512, 256]
SAC_ACTIVATION = 'relu'
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Measurement parameters
TIMING_RUNS = 2000
WARMUP_RUNS = 200
BATCH_SIZE = 1  # For runtime inference

# ==================================================================
# --- MODEL DEFINITIONS ---
# (Copied from sac_policy_network.py)
# ==================================================================

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

    def _get_activation(self, activation_name: str) -> nn.Module:
        if activation_name == "relu": return nn.ReLU
        elif activation_name == "tanh": return nn.Tanh
        elif activation_name == "elu": return nn.ELU
        else: raise ValueError(f"Unsupported activation: {activation_name}")

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

class Actor(nn.Module):
    """
    SAC Actor (Policy) Network.
    """
    def __init__(
        self,
        state_dim: int = (N_JOINTS * 2) * 2, # (predicted_state + remote_state)
        action_dim: int = N_JOINTS,
        hidden_dims: list = SAC_MLP_HIDDEN_DIMS,
        activation: str = SAC_ACTIVATION
    ):
        super().__init__()
        self.activation_fn = self._get_activation(activation)

        layers = []
        last_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(self.activation_fn())
            last_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.fc_mean = nn.Linear(last_dim, action_dim)
        self.fc_log_std = nn.Linear(last_dim, action_dim)

        self._initialize_weights()

        # Action scaling from [-1, 1] to actual torque compensation range
        self.register_buffer(
            'action_scale',
            torch.tensor(MAX_TORQUE_COMPENSATION, dtype=torch.float32)
        )
        self.register_buffer(
            'action_bias',
            torch.tensor(0.0, dtype=torch.float32)
        )

    def _get_activation(self, activation_name: str) -> nn.Module:
        if activation_name == "relu": return nn.ReLU
        elif activation_name == "tanh": return nn.Tanh
        elif activation_name == "elu": return nn.ELU
        else: raise ValueError(f"Unsupported activation: {activation_name}")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(state)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self,
               state: torch.Tensor,
               deterministic: bool = False
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal_dist = Normal(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = normal_dist.rsample()

        tanh_action = torch.tanh(raw_action)

        log_prob = normal_dist.log_prob(raw_action)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        scaled_action = self.action_scale * tanh_action + self.action_bias

        return scaled_action, log_prob, raw_action

# ==================================================================
# --- HELPER FUNCTIONS ---
# ==================================================================

def load_models(device: torch.device) -> Tuple[StateEstimator, Actor]:
    """Loads the models from disk."""
    print("Loading models...")
    
    # Load Estimator
    if not os.path.exists(ESTIMATOR_PATH):
        raise FileNotFoundError(f"Estimator file not found: {ESTIMATOR_PATH}")
    estimator = StateEstimator(
        input_dim=N_JOINTS * 2,
        hidden_dim=RNN_HIDDEN_DIM,
        num_layers=RNN_NUM_LAYERS,
        output_dim=N_JOINTS * 2
    )
    estimator_data = torch.load(ESTIMATOR_PATH, map_location=device)
    # Note: The estimator_best.pth from pretraining saves the dict directly
    estimator.load_state_dict(estimator_data.get('state_estimator_state_dict', estimator_data))
    estimator.to(device).eval()
    print(f"  ✓ StateEstimator loaded from {ESTIMATOR_PATH}")

    # Load Actor
    if not os.path.exists(POLICY_PATH):
        raise FileNotFoundError(f"Policy file not found: {POLICY_PATH}")
    actor = Actor(
        state_dim=(N_JOINTS * 2) * 2,
        action_dim=N_JOINTS,
        hidden_dims=SAC_MLP_HIDDEN_DIMS,
        activation=SAC_ACTIVATION
    )
    policy_data = torch.load(POLICY_PATH, map_location=device, weights_only=False)
    actor.load_state_dict(policy_data['actor_state_dict'])
    actor.to(device).eval()
    print(f"  ✓ Actor loaded from {POLICY_PATH}")
    
    return estimator, actor

def print_stats(test_name: str, times_ms: np.ndarray):
    """Prints formatted statistics."""
    print(f"\n--- {test_name} ---")
    print(f"  Total Runs: {len(times_ms)}")
    print(f"  Mean:       {np.mean(times_ms):.4f} ms")
    print(f"  Std Dev:    {np.std(times_ms):.4f} ms")
    print(f"  Median (p50): {np.percentile(times_ms, 50):.4f} ms")
    print(f"  p95:        {np.percentile(times_ms, 95):.4f} ms")
    print(f"  p99:        {np.percentile(times_ms, 99):.4f} ms")
    print(f"  Min:        {np.min(times_ms):.4f} ms")
    print(f"  Max:        {np.max(times_ms):.4f} ms")
    print(f"  FPS (Mean): {1000.0 / np.mean(times_ms):.2f} Hz")

# ==================================================================
# --- MAIN EXECUTION ---
# ==================================================================

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Timing {TIMING_RUNS} runs after {WARMUP_RUNS} warmup runs...")
    
    try:
        estimator, actor = load_models(device)
    except Exception as e:
        print(f"\n[ERROR] Failed to load models: {e}")
        print("Please check your paths and config values.")
        exit(1)

    # --- 1. Create Dummy Data ---
    print("\nCreating dummy input data...")
    state_dim = N_JOINTS * 2  # 14
    actor_input_dim = state_dim * 2  # 28
    
    # For stateful, single-step LSTM (as used in sac_training_algorithm.py)
    dummy_delayed_obs_step = torch.randn(BATCH_SIZE, 1, state_dim).to(device)
    dummy_hidden_state = estimator.init_hidden_state(BATCH_SIZE, device)
    
    # For Actor
    dummy_actor_input = torch.randn(BATCH_SIZE, actor_input_dim).to(device)

    # For End-to-End
    dummy_remote_state = torch.randn(BATCH_SIZE, state_dim).to(device)


    # --- 2. Test 1: StateEstimator (Stateful, Single-Step) ---
    # This simulates the runtime usage
    
    # Warmup
    print("Warming up StateEstimator...")
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            _, dummy_hidden_state = estimator(dummy_delayed_obs_step, dummy_hidden_state)
            
    # Timing
    times = []
    for _ in range(TIMING_RUNS):
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _, dummy_hidden_state = estimator(dummy_delayed_obs_step, dummy_hidden_state)
        
        if device.type == 'cuda': torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0) # Store in milliseconds
        
    print_stats("Test 1: StateEstimator (Stateful, B=1, Seq=1)", np.array(times))

    
    # --- 3. Test 2: Actor (Policy) ---
    
    # Warmup
    print("\nWarming up Actor...")
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            _ = actor.sample(dummy_actor_input, deterministic=True)
            
    # Timing
    times = []
    for _ in range(TIMING_RUNS):
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = actor.sample(dummy_actor_input, deterministic=True)
        
        if device.type == 'cuda': torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
        
    print_stats("Test 2: Actor (B=1)", np.array(times))

    
    # --- 4. Test 3: End-to-End Pipeline ---
    # This is the most critical metric: (Estimator -> Actor)
    
    # Reset hidden state for a fair test
    dummy_hidden_state = estimator.init_hidden_state(BATCH_SIZE, device)
    
    # Warmup
    print("\nWarming up End-to-End Pipeline...")
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            pred_state, dummy_hidden_state = estimator(dummy_delayed_obs_step, dummy_hidden_state)
            actor_input = torch.cat([pred_state, dummy_remote_state], dim=1)
            _ = actor.sample(actor_input, deterministic=True)
            
    # Timing
    times = []
    for _ in range(TIMING_RUNS):
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            # Step 1: Estimator
            pred_state, dummy_hidden_state = estimator(dummy_delayed_obs_step, dummy_hidden_state)
            
            # Step 2: Actor
            actor_input = torch.cat([pred_state, dummy_remote_state], dim=1)
            _ = actor.sample(actor_input, deterministic=True)
        
        if device.type == 'cuda': torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
        
    print_stats(f"Test 3: End-to-End Pipeline (B=1)", np.array(times))