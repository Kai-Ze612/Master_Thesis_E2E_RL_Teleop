"""
SAC Policy Network.

This is an E2E model, with LSTM encoder intergration.

This script is a forward training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Normal

import E2E_Teleoperation.config.robot_config as cfg


class SharedLSTMEncoder(nn.Module):
    """
    Shared LSTM encoder used by both Actor and Critic.
    """
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=cfg.ESTIMATOR_STATE_DIM,  # 15
            hidden_size=cfg.RNN_HIDDEN_DIM,       # 256
            num_layers=cfg.RNN_NUM_LAYERS,
            batch_first=True
        )
        
        # Auxiliary Head (Predicts current state for Loss)
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.RNN_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, cfg.ESTIMATOR_OUTPUT_DIM)  # 14: Predicts q, qd
        )
    
    def forward(
        self, 
        history_seq: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the Shared LSTM Encoder.
        
        Input: history_sequence (15D * sequence length)
        Ouput: 14D current local robot state
        """
        lstm_out, next_hidden = self.lstm(history_seq, hidden)
        lstm_features = lstm_out[:, -1, :]  # Last hidden state
        pred_state = self.aux_head(lstm_features)
        return lstm_features, pred_state, next_hidden

class JointActor(nn.Module):
    def __init__(
        self,
        shared_encoder: SharedLSTMEncoder,
        action_dim: int = cfg.N_JOINTS,
        hidden_dims: list = cfg.MLP_HIDDEN_DIMS,
    ):
        super().__init__()
        
        # Use shared encoder
        self.encoder = shared_encoder
        
        # Calculate dimensions based on config
        self.robot_state_dim = cfg.ROBOT_STATE_DIM
        self.target_hist_dim = cfg.TARGET_HISTORY_DIM
        self.robot_hist_dim = cfg.ROBOT_HISTORY_DIM
        
        # Policy Backbone Input
        policy_input_dim = cfg.RNN_HIDDEN_DIM + self.robot_state_dim + self.robot_hist_dim
        
        layers = []
        last_dim = policy_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        
        self.fc_mean = nn.Linear(last_dim, action_dim)
        self.fc_log_std = nn.Linear(last_dim, action_dim)
        
        # Initialize small weights to prevent saturation at start
        nn.init.uniform_(self.fc_mean.weight, -1e-3, 1e-3)
        nn.init.constant_(self.fc_mean.bias, 0.0)
        
        self.register_buffer('action_scale', torch.tensor(cfg.MAX_ACTION_TORQUE))
        self.register_buffer('action_bias', torch.tensor(0.0))
        
    def forward(self, obs_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Slice inputs
        remote_state = obs_flat[:, :self.robot_state_dim]
        robot_history = obs_flat[:, self.robot_state_dim : -self.target_hist_dim]
        target_history = obs_flat[:, -self.target_hist_dim:]
        
        # 2. Encoder Pass
        batch_size = obs_flat.shape[0]
        history_seq = target_history.view(batch_size, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
        lstm_features, pred_state, _ = self.encoder(history_seq)
        
        # 3. Policy Pass
        policy_input = torch.cat([lstm_features.detach(), remote_state, robot_history], dim=1)
        x = self.backbone(policy_input)
        
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, cfg.LOG_STD_MIN, cfg.LOG_STD_MAX)
        
        return mean, log_std, pred_state

    def sample(self, obs_flat: torch.Tensor, deterministic: bool = False):
        mean, log_std, pred_state = self.forward(obs_flat)
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
        
        return scaled_action, log_prob, raw_action, pred_state


class JointCritic(nn.Module):
    def __init__(
        self, 
        shared_encoder: SharedLSTMEncoder,
        action_dim: int = cfg.N_JOINTS, 
        hidden_dims: list = cfg.MLP_HIDDEN_DIMS
    ):
        super().__init__()
        self.encoder = shared_encoder
        
        self.robot_state_dim = cfg.ROBOT_STATE_DIM
        self.target_hist_dim = cfg.TARGET_HISTORY_DIM
        self.robot_hist_dim = cfg.ROBOT_HISTORY_DIM
        
        input_dim = cfg.RNN_HIDDEN_DIM + self.robot_state_dim + self.robot_hist_dim + action_dim
        
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.register_buffer('action_scale', torch.tensor(cfg.MAX_ACTION_TORQUE))

    def forward(self, obs_flat: torch.Tensor, action: torch.Tensor):
        remote_state = obs_flat[:, :self.robot_state_dim]
        robot_history = obs_flat[:, self.robot_state_dim : -self.target_hist_dim]
        target_history = obs_flat[:, -self.target_hist_dim:]
        
        batch_size = obs_flat.shape[0]
        history_seq = target_history.view(batch_size, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
        lstm_features, _, _ = self.encoder(history_seq)

        # Normalize Action (Full Torque -> [-1, 1])
        action_normalized = action / self.action_scale
        
        xu = torch.cat([lstm_features, remote_state, robot_history, action_normalized], dim=1)
        return self.q1_net(xu), self.q2_net(xu)

def create_actor_critic(device: str = 'cuda'):
    shared_encoder = SharedLSTMEncoder().to(device)
    actor = JointActor(shared_encoder, action_dim=cfg.N_JOINTS).to(device)
    critic = JointCritic(shared_encoder, action_dim=cfg.N_JOINTS).to(device)
    return shared_encoder, actor, critic