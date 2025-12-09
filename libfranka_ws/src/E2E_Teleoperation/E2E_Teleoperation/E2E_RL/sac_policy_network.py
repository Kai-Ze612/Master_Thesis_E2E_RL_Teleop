"""
Jointly training of State Estimator and SAC policy network.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Normal

import E2E_Teleoperation.config.robot_config as cfg


class JointActor(nn.Module):
    """
    Joint Actor: Contains internal LSTM + MLP Policy
    """
    def __init__(
        self,
        action_dim: int = cfg.N_JOINTS,
        hidden_dims: list = cfg.SAC_MLP_HIDDEN_DIMS,
    ):
        super().__init__()
        
        # 1. Internal LSTM
        self.lstm = nn.LSTM(
            input_size=cfg.ESTIMATOR_STATE_DIM, # 15
            hidden_size=cfg.RNN_HIDDEN_DIM,     # 256
            num_layers=cfg.RNN_NUM_LAYERS,
            batch_first=True
        )
        
        # Auxiliary Head (Predicts current state for Loss)
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.RNN_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 14) # Predicts q, qd
        )
        
        # 2. Policy Backbone (The "Hand")
        # Input: LSTM Features (256) + Current Remote State (14)
        policy_input_dim = cfg.RNN_HIDDEN_DIM + (cfg.N_JOINTS * 2)
        
        layers = []
        last_dim = policy_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        
        self.fc_mean = nn.Linear(last_dim, action_dim)
        self.fc_log_std = nn.Linear(last_dim, action_dim)
        
        self.register_buffer('action_scale', torch.tensor(cfg.TORQUE_LIMITS))
        self.register_buffer('action_bias', torch.tensor(0.0))

    def forward(self, obs_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: Flat Observation Vector
        Output: Mean, LogStd, PREDICTED_STATE (for Aux Loss)
        """
        
        remote_state = obs_flat[:, :14]
        history_flat = obs_flat[:, 14:]
        
        # Reshape history to (Batch, Seq_Len, 15)
        batch_size = obs_flat.shape[0]
        history_seq = history_flat.view(batch_size, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
        
        # 2. Run LSTM
        lstm_out, _ = self.lstm(history_seq)
        lstm_features = lstm_out[:, -1, :] # Last hidden state
        
        # 3. Aux Task: Predict Current State
        pred_state = self.aux_head(lstm_features)
        
        # 4. Policy: Concatenate Features + Remote State
        policy_input = torch.cat([lstm_features, remote_state], dim=1)
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
        
        # Return pred_state too!
        return scaled_action, log_prob, raw_action, pred_state


class JointCritic(nn.Module):
    def __init__(self, action_dim=cfg.N_JOINTS, hidden_dims=cfg.SAC_MLP_HIDDEN_DIMS):
        super().__init__()
        # Similar LSTM structure...
        self.lstm = nn.LSTM(cfg.ESTIMATOR_STATE_DIM, cfg.RNN_HIDDEN_DIM, cfg.RNN_NUM_LAYERS, batch_first=True)
        
        # Q1
        input_dim = cfg.RNN_HIDDEN_DIM + (cfg.N_JOINTS * 2) + action_dim
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs_flat, action):
        # Unpack & LSTM
        remote_state = obs_flat[:, :14]
        history_flat = obs_flat[:, 14:]
        batch_size = obs_flat.shape[0]
        history_seq = history_flat.view(batch_size, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
        
        lstm_out, _ = self.lstm(history_seq)
        features = lstm_out[:, -1, :]
        
        xu = torch.cat([features, remote_state, action], dim=1)
        return self.q1_net(xu), self.q2_net(xu)