"""
Define the Network Architecture.
1. LSTM State Estimator
2. SAC Actor Network
3. SAC Critic Network

"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Normal

import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg


class StateEstimator(nn.Module):
    """
    Step-Based Autoregressive LSTM.
    
    CRITICAL DIMENSION NOTE:
    - Input: 15D (7 Joint Pos + 7 Joint Vel + 1 Delay Scalar)
    - Output: 14D (7 Joint Pos + 7 Joint Vel)
    
    The 'Delay Scalar' must be manually managed and decremented during 
    autoregressive inference loops.
    """
    def __init__(
        self,
        input_dim_total: int = cfg.ESTIMATOR_STATE_DIM,  # Default: 15
        hidden_dim: int = cfg.RNN_HIDDEN_DIM,
        num_layers: int = cfg.RNN_NUM_LAYERS,
        output_dim: int = cfg.ESTIMATOR_OUTPUT_DIM,  # Default: 14
    ):
        
        super().__init__()
        
        self.input_dim_total = input_dim_total
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim_total,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Prediction head (Predicts next step state)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for processing a full sequence (Training Mode).
        Returns prediction for the LAST step in the sequence.
        """
        # x shape: (Batch, Seq_Len, 15)
        lstm_out, _ = self.lstm(x)
        
        # Take the last hidden state from the sequence
        last_hidden = lstm_out[:, -1, :]
        
        # Predict the next state (14D)
        pred_state = self.fc(last_hidden)
        
        return pred_state, last_hidden

    def forward_step(
        self, 
        x_step: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step inference for Autoregressive Loop (Deployment Mode).
        
        Args:
            x_step: Tensor of shape (Batch, 1, 15). 
                    MUST INCLUDE THE DELAY SCALAR AS THE 15th FEATURE.
            hidden_state: Tuple (h_n, c_n) from previous step.
            
        Returns:
            pred_state: (Batch, 1, 14) -> Next Robot State
            new_hidden: Updated hidden state for next step
        """
        # Safety Check for Dimensions
        if x_step.shape[-1] != self.input_dim_total:
            raise ValueError(
                f"StateEstimator Dimension Error: Expected input dim {self.input_dim_total} (14 State + 1 Delay), "
                f"but got {x_step.shape[-1]}. Did you forget to concatenate the delay scalar?"
            )

        # LSTM forward with hidden state
        # lstm_out: (Batch, 1, Hidden_Dim)
        lstm_out, new_hidden = self.lstm(x_step, hidden_state)
        
        # Predict next state
        pred_state = self.fc(lstm_out) # (Batch, 1, 14)
        
        return pred_state, new_hidden


class Actor(nn.Module):
    """
    SAC Actor Network (Stochastic).
    """
    def __init__(
        self,
        state_dim: int = cfg.OBS_DIM,
        action_dim: int = cfg.N_JOINTS,
        hidden_dims: list = cfg.SAC_MLP_HIDDEN_DIMS,
        activation: str = cfg.SAC_ACTIVATION
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
        self.register_buffer('action_scale', torch.tensor(cfg.MAX_TORQUE_COMPENSATION, dtype=torch.float32))
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
        log_std = torch.clamp(log_std, cfg.LOG_STD_MIN, cfg.LOG_STD_MAX)
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
        state_dim: int = cfg.OBS_DIM,
        action_dim: int = cfg.N_JOINTS,
        hidden_dims: list = cfg.SAC_MLP_HIDDEN_DIMS,
        activation: str = cfg.SAC_ACTIVATION
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