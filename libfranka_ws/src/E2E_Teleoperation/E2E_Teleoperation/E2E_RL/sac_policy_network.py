"""
This module defines the policy network architecture used in the
End-to-End Teleoperation system based on Soft Actor-Critic (SAC).

Three main components are implemented:
1. Autoregressive LSTM Encoder for temporal state estimation.
2. Inverse Dynamics Actor Network for torque prediction.
3. SAC Critic Network operating on latent features.

The stage of training determines which components are active:
- Stage 1: Only the LSTM Encoder is used for state prediction.
- Stage 2: The Actor Network is added for inverse dynamics control.
- Stage 3: The Critic Network is included for SAC training.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import E2E_Teleoperation.config.robot_config as cfg

class LSTM(nn.Module):
    """
    Stage 1 Component: Temporal Encoder
    Input: Delayed History Sequence
    Output: Predicted State (14-dim: 7 Pos + 7 Vel)
    """
    def __init__(self):
        super().__init__()
        
        # 1. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=cfg.ROBOT.ESTIMATOR_INPUT_DIM, # 15
            hidden_size=cfg.ROBOT.RNN_HIDDEN_DIM,     # 256
            num_layers=cfg.ROBOT.RNN_NUM_LAYERS,      # 3
            batch_first=True
        )
        
        # 2. State Predictor Head (MLP)
        # FIX: Matches proposal (256 -> 256 -> 14)
        self.predictor = nn.Sequential(
            nn.Linear(cfg.ROBOT.RNN_HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, cfg.ROBOT.ROBOT_STATE_DIM) # 14
        )

    def forward(self, history, hidden=None):
        out, hidden = self.lstm(history, hidden)
        feat = out[:, -1, :] 
        pred_state = self.predictor(feat)
        return feat, pred_state, hidden


class JointActor(nn.Module):
    """
    Stage 2 & 3 Component: Inverse Dynamics Policy
    Input: [Predicted_State(14), Desired_State(14)] = 28 dims
    Output: Torque(7)
    """
    LOG_STD_MIN = -10.0
    LOG_STD_MAX = 2.0
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.input_dim = cfg.ROBOT.ROBOT_STATE_DIM * 2 
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        
        self.mu = nn.Linear(256, cfg.ROBOT.N_JOINTS)
        self.log_std = nn.Linear(256, cfg.ROBOT.N_JOINTS)
        self.scale = torch.tensor(cfg.ROBOT.TORQUE_LIMITS)

    def forward(self, obs, hidden=None):
        # 1. Parse Observation
        idx_rem = cfg.ROBOT.ROBOT_STATE_DIM
        idx_hist = idx_rem + cfg.ROBOT.ROBOT_HISTORY_DIM
        target_hist = obs[:, idx_hist:]
        
        target_seq = target_hist.view(-1, cfg.ROBOT.RNN_SEQ_LEN, cfg.ROBOT.ESTIMATOR_INPUT_DIM)
        desired_state = target_seq[:, -1, :cfg.ROBOT.ROBOT_STATE_DIM]

        # 2. Get Predicted State from Encoder
        feat, pred_state, next_hidden = self.encoder(target_seq, hidden)

        # 3. Concatenate [Predicted, Desired]
        x = torch.cat([pred_state, desired_state], dim=1)
        
        x = self.net(x)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mu, log_std, pred_state, next_hidden, feat

    def sample(self, obs, hidden=None):
        mu, log_std, pred_state, next_hidden, feat = self.forward(obs, hidden)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample() 
        y_t = torch.tanh(x_t)
        action = y_t * self.scale.to(obs.device)
        
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob, pred_state, next_hidden, feat


class JointCritic(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.input_dim = cfg.ROBOT.RNN_HIDDEN_DIM + cfg.ROBOT.N_JOINTS
        
        self.q1 = nn.Sequential(nn.Linear(self.input_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(self.input_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, feat, action):
        xu = torch.cat([feat, action], dim=1)
        return self.q1(xu), self.q2(xu)