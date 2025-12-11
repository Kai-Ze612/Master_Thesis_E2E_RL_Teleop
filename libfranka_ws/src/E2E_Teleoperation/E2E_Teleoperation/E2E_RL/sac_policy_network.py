"""
SAC Policy Network with Shared LSTM Encoder.

Key Changes:
1. Critic uses the SAME LSTM as Actor (shared encoder)
2. This ensures consistent state representations
3. Proper weight sharing reduces parameters and improves learning
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
    This ensures consistent state representations.
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
    
    def forward(self, history_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history_seq: (batch, seq_len, 15) normalized history sequence
            
        Returns:
            lstm_features: (batch, 256) last hidden state
            pred_state: (batch, 14) predicted current state
        """
        lstm_out, _ = self.lstm(history_seq)
        lstm_features = lstm_out[:, -1, :]  # Last hidden state
        pred_state = self.aux_head(lstm_features)
        return lstm_features, pred_state

class JointActor(nn.Module):
    def __init__(
        self,
        shared_encoder: SharedLSTMEncoder,
        action_dim: int = cfg.N_JOINTS,
        hidden_dims: list = cfg.SAC_MLP_HIDDEN_DIMS,
    ):
        super().__init__()
        
        # Use shared encoder
        self.encoder = shared_encoder
        
        # --- [MODIFIED SECTION START] ---
        # Calculate dimensions based on config
        self.robot_state_dim = 14  # 7 q + 7 qd
        self.target_hist_dim = cfg.RNN_SEQUENCE_LENGTH * cfg.ESTIMATOR_STATE_DIM # 80 * 15 = 1200
        self.robot_hist_dim = cfg.RNN_SEQUENCE_LENGTH * self.robot_state_dim     # 80 * 14 = 1120
        
        # Policy Backbone Input: 
        # LSTM Features (256) + Current State (14) + Robot History (1120)
        policy_input_dim = cfg.RNN_HIDDEN_DIM + self.robot_state_dim + self.robot_hist_dim
        # --- [MODIFIED SECTION END] ---
        
        layers = []
        last_dim = policy_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        
        self.fc_mean = nn.Linear(last_dim, action_dim)
        self.fc_log_std = nn.Linear(last_dim, action_dim)
        
        self.register_buffer('action_scale', torch.tensor(cfg.MAX_TORQUE_COMPENSATION))
        self.register_buffer('action_bias', torch.tensor(0.0))
        
    def forward(self, obs_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: Flat Observation Vector [Current State | Robot History | Target History]
        """
        # --- [MODIFIED SECTION START] ---
        # 1. Slice the inputs
        # Current State: First 14
        remote_state = obs_flat[:, :self.robot_state_dim]
        
        # Robot History: Middle Chunk (14 to -1200)
        # This gives the actor the "eyes" to see its own past oscillations
        robot_history = obs_flat[:, self.robot_state_dim : -self.target_hist_dim]
        
        # Target History: Last Chunk (Last 1200) -> Goes to Encoder
        target_history = obs_flat[:, -self.target_hist_dim:]
        
        # 2. Process Target History through Shared Encoder
        batch_size = obs_flat.shape[0]
        history_seq = target_history.view(batch_size, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
        
        # Get LSTM features (z)
        lstm_features, pred_state = self.encoder(history_seq)
        
        # 3. Concatenate EVERYTHING for the Policy MLP
        # [Embedding (z)] + [Current State] + [Robot History (Momentum)]
        policy_input = torch.cat([lstm_features.detach(), remote_state, robot_history], dim=1)
        
        x = self.backbone(policy_input)
        # --- [MODIFIED SECTION END] ---
        
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, cfg.LOG_STD_MIN, cfg.LOG_STD_MAX)
        
        return mean, log_std, pred_state

    # --- [THIS METHOD WAS MISSING] ---
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
        hidden_dims: list = cfg.SAC_MLP_HIDDEN_DIMS
    ):
        super().__init__()
        
        # Use the SAME shared encoder as actor
        self.encoder = shared_encoder
        
        # --- [MODIFIED SECTION START] ---
        self.robot_state_dim = 14
        self.target_hist_dim = cfg.RNN_SEQUENCE_LENGTH * cfg.ESTIMATOR_STATE_DIM
        self.robot_hist_dim = cfg.RNN_SEQUENCE_LENGTH * self.robot_state_dim
        
        # Q-network input: 
        # LSTM features (256) + Current (14) + Robot Hist (1120) + Action (7)
        input_dim = cfg.RNN_HIDDEN_DIM + self.robot_state_dim + self.robot_hist_dim + action_dim
        # --- [MODIFIED SECTION END] ---
        
        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs_flat: torch.Tensor, action: torch.Tensor):
        # --- [MODIFIED SECTION START] ---
        # 1. Slice inputs (Same logic as Actor)
        remote_state = obs_flat[:, :self.robot_state_dim]
        robot_history = obs_flat[:, self.robot_state_dim : -self.target_hist_dim]
        target_history = obs_flat[:, -self.target_hist_dim:]
        
        # 2. Encoder Pass
        batch_size = obs_flat.shape[0]
        history_seq = target_history.view(batch_size, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
        
        # Get LSTM features
        lstm_features, _ = self.encoder(history_seq)
        
        # 3. Concatenate for Q-Networks
        # Note: We do NOT detach LSTM features here during updates, usually
        xu = torch.cat([lstm_features, remote_state, robot_history, action], dim=1)
        # --- [MODIFIED SECTION END] ---
        
        return self.q1_net(xu), self.q2_net(xu)


# ============================================================================
# Factory function to create Actor and Critic with shared encoder
# ============================================================================
def create_actor_critic(device: str = 'cuda'):
    """
    Create Actor and Critic networks with a shared LSTM encoder.
    
    Returns:
        shared_encoder: The shared LSTM encoder
        actor: JointActor using the shared encoder
        critic: JointCritic using the shared encoder
    """
    shared_encoder = SharedLSTMEncoder().to(device)
    actor = JointActor(shared_encoder, action_dim=cfg.N_JOINTS).to(device)
    critic = JointCritic(shared_encoder, action_dim=cfg.N_JOINTS).to(device)
    
    return shared_encoder, actor, critic