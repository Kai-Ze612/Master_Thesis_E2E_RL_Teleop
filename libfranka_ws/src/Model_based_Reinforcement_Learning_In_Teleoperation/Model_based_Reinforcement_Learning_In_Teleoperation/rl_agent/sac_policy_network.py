"""
Define the Network Architecture of:
1. LSTM state predicter
2. Actor: The SAC stochastic policy, which takes the *predicted* state
    and outputs a residual torque compensation.
3. Critic: The SAC Q-function, which evaluates the value of a
    (predicted_state, action) pair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
from typing import Tuple, Optional, Union

# Custom imports
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    MAX_TORQUE_COMPENSATION,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    SAC_MLP_HIDDEN_DIMS,
    SAC_ACTIVATION,
    LOG_STD_MIN,
    LOG_STD_MAX,
)

# LSTM State Estimator Architecture
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

class Critic(nn.Module):
    """
    SAC Critic (Q-Function) Network.
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

        # Q1 Network
        layers_q1 = []
        last_dim_q1 = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers_q1.append(nn.Linear(last_dim_q1, hidden_dim))
            layers_q1.append(self.activation_fn())
            last_dim_q1 = hidden_dim
        layers_q1.append(nn.Linear(last_dim_q1, 1))
        self.q1_net = nn.Sequential(*layers_q1)

        # Q2 Network
        layers_q2 = []
        last_dim_q2 = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers_q2.append(nn.Linear(last_dim_q2, hidden_dim))
            layers_q2.append(self.activation_fn())
            last_dim_q2 = hidden_dim
        layers_q2.append(nn.Linear(last_dim_q2, 1))
        self.q2_net = nn.Sequential(*layers_q2)

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

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2