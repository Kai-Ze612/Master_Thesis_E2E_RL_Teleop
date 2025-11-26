"""
Define the Network Architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Normal

# Import configs for Actor/Critic
# We use hardcoded values for StateEstimator to ensure it matches the checkpoint
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    OBS_DIM,
    MAX_TORQUE_COMPENSATION,
    SAC_MLP_HIDDEN_DIMS,
    SAC_ACTIVATION,
    LOG_STD_MIN,
    LOG_STD_MAX,
    RNN_NUM_LAYERS,
    RNN_HIDDEN_DIM,
)

class StateEstimator(nn.Module):
    """
    LSTM-based State Estimator.
    Structure matches the pre-trained checkpoint exactly:
    - Input: 15 (7 q + 7 qd + 1 delay)
    - Hidden: 256
    - Head: 'fc' (not 'prediction_head')
    - No LayerNorm ('input_ln')
    """
    def __init__(
        self,
        input_dim_total: int = 15,  # FIXED: Checkpoint expects 15 inputs
        hidden_dim: int = RNN_HIDDEN_DIM,      # FIXED: Checkpoint expects 256 hidden size
        num_layers: int = RNN_NUM_LAYERS,
        output_dim: int = 14,
    ):
        super().__init__()
        
        # Store dimensions for later use
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. LSTM Core
        self.lstm = nn.LSTM(
            input_size=input_dim_total,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 2. Prediction Head (Must be named 'fc')
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # NOTE: Removed 'input_ln' as it does not exist in the checkpoint

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for full sequence.
        
        Args:
            x: (Batch, Seq_Len, 15) - Input sequence
            
        Returns:
            residual: (Batch, 14) - Predicted state residual
            last_hidden: (Batch, hidden_dim) - Last hidden state output
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output
        last_hidden = lstm_out[:, -1, :]
        
        # Predict residual
        residual = self.fc(last_hidden)
        
        return residual, last_hidden

    def forward_step(
        self, 
        x_step: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step forward for autoregressive inference.
        
        This method maintains hidden state across multiple prediction steps,
        which is essential for proper autoregressive rollout.
        
        Args:
            x_step: (Batch, 1, 15) - Single timestep input
            hidden_state: Optional tuple of (h, c) from previous step
                          Each has shape (num_layers, batch, hidden_dim)
        
        Returns:
            residual: (Batch, 14) - Predicted state residual
            new_hidden: Tuple of (h, c) for next step
        """
        # LSTM forward with hidden state
        lstm_out, new_hidden = self.lstm(x_step, hidden_state)
        
        # Predict residual from the output (only one timestep)
        residual = self.fc(lstm_out[:, -1, :])
        
        return residual, new_hidden


class Actor(nn.Module):
    """
    SAC Actor Network (Stochastic).
    """
    def __init__(
        self,
        state_dim: int = OBS_DIM,         # Defaults to Config (112D)
        action_dim: int = N_JOINTS,       # Defaults to Config (7D)
        hidden_dims: list = SAC_MLP_HIDDEN_DIMS,
        activation: str = SAC_ACTIVATION
    ):
        super().__init__()
        self.activation_fn = self._get_activation(activation)
        
        # Build MLP Backbone
        layers = []
        last_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(self.activation_fn())
            last_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Output Heads
        self.fc_mean = nn.Linear(last_dim, action_dim)
        self.fc_log_std = nn.Linear(last_dim, action_dim)
        
        self._initialize_weights()
        
        # register buffers to handle device movement automatically
        self.register_buffer('action_scale', torch.tensor(MAX_TORQUE_COMPENSATION, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor(0.0, dtype=torch.float32))

    def _get_activation(self, activation_name: str) -> nn.Module:
        if activation_name.lower() == "relu": return nn.ReLU
        elif activation_name.lower() == "tanh": return nn.Tanh
        elif activation_name.lower() == "elu": return nn.ELU
        elif activation_name.lower() == "mish": return nn.Mish
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
        # Clamp log_std to maintain numerical stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal_dist = Normal(mean, std)
        
        if deterministic:
            raw_action = mean
        else:
            raw_action = normal_dist.rsample() # Reparameterization trick
            
        # Apply Tanh to squash to [-1, 1]
        tanh_action = torch.tanh(raw_action)
        
        # Calculate Log Prob
        log_prob = normal_dist.log_prob(raw_action)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Scale to physical torque limits
        scaled_action = self.action_scale * tanh_action + self.action_bias
        
        return scaled_action, log_prob, raw_action


class Critic(nn.Module):
    """
    SAC Critic Network (Twin Q-Networks).
    """
    def __init__(
        self,
        state_dim: int = OBS_DIM,         # Defaults to Config (112D)
        action_dim: int = N_JOINTS,       # Defaults to Config (7D)
        hidden_dims: list = SAC_MLP_HIDDEN_DIMS,
        activation: str = SAC_ACTIVATION
    ):
        super().__init__()
        self.activation_fn = self._get_activation(activation)
        
        # Q1 Architecture
        layers_q1 = []
        last_dim_q1 = state_dim + action_dim
        for h_dim in hidden_dims:
            layers_q1.append(nn.Linear(last_dim_q1, h_dim))
            layers_q1.append(self.activation_fn())
            last_dim_q1 = h_dim
        layers_q1.append(nn.Linear(last_dim_q1, 1))
        self.q1_net = nn.Sequential(*layers_q1)

        # Q2 Architecture (Twin)
        layers_q2 = []
        last_dim_q2 = state_dim + action_dim
        for h_dim in hidden_dims:
            layers_q2.append(nn.Linear(last_dim_q2, h_dim))
            layers_q2.append(self.activation_fn())
            last_dim_q2 = h_dim
        layers_q2.append(nn.Linear(last_dim_q2, 1))
        self.q2_net = nn.Sequential(*layers_q2)
        
        self._initialize_weights()

    def _get_activation(self, activation_name: str) -> nn.Module:
        if activation_name.lower() == "relu": return nn.ReLU
        elif activation_name.lower() == "tanh": return nn.Tanh
        elif activation_name.lower() == "elu": return nn.ELU
        elif activation_name.lower() == "mish": return nn.Mish
        else: raise ValueError(f"Unsupported activation: {activation_name}")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2