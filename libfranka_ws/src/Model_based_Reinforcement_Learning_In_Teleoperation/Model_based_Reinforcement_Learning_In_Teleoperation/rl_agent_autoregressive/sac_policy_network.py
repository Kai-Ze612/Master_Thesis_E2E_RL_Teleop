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
    Physics-Informed Residual LSTM (Euler Integration).
    
    Instead of predicting 'Delta Position' (which is tiny and causes staircasing),
    we predict 'Normalized Velocity' (which is large).
    
    Equation:
        v_pred = Network(state)
        next_state = state + v_pred * dt_integration_constant
    """
    def __init__(
        self,
        input_dim_total: int = cfg.ESTIMATOR_STATE_DIM, # 15
        hidden_dim: int = cfg.RNN_HIDDEN_DIM,
        num_layers: int = cfg.RNN_NUM_LAYERS,
        output_dim: int = cfg.ESTIMATOR_OUTPUT_DIM,     # 14
    ):
        super().__init__()
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim_total,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # We use Mish or LeakyReLU to prevent "dead neurons" (strict zeros)
        self.activation = nn.Mish() 
        
        # Network outputs [Normalized Velocity, Normalized Acceleration]
        # Magnitude is ~1.0, so gradients flow easily.
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            self.activation,
            nn.Linear(256, output_dim)
        )
        
        # Integration Constant: Effectively dt scaled by normalization stats.
        # At 200Hz, this is roughly 0.01. 
        # Making this a parameter allows the network to fine-tune the time constant.
        self.dt_scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training Forward Pass"""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # 1. Predict Normalized Derivatives (Velocity/Accel)
        derivatives = self.fc(last_hidden)
        
        # 2. Get Previous State
        # x is [Position (7), Velocity (7), Delay (1)]
        prev_state = x[:, -1, :self.output_dim]
        
        # 3. Euler Integration: State_New = State_Old + Deriv * dt
        # This forces the network to output non-zero derivatives to minimize loss.
        pred_state = prev_state + derivatives * self.dt_scale
        
        return pred_state, last_hidden

    def forward_step(
        self, 
        x_step: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Inference Forward Pass"""
        lstm_out, new_hidden = self.lstm(x_step, hidden_state)
        
        # 1. Predict Derivatives
        derivatives = self.fc(lstm_out)
        
        # 2. Get Previous State
        prev_state = x_step[:, :, :self.output_dim]
        
        # 3. Euler Integration
        pred_state = prev_state + derivatives * self.dt_scale
        
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