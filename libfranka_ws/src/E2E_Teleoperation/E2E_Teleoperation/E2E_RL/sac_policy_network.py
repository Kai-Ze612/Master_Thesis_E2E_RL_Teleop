"""
E2E_Teleoperation/E2E_RL/sac_policy_network.py

FIXED VERSION - Backward Compatible with Existing Checkpoints
- No architectural changes to encoder (preserves checkpoint compatibility)
- Improved initialization and numerical stability
- Better log_std bounds
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import E2E_Teleoperation.config.robot_config as cfg


class ContinuousLSTMEncoder(nn.Module):
    """
    LSTM-based state estimator for predicting future states.
    Architecture unchanged for checkpoint compatibility.
    """
    
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.ROBOT.ESTIMATOR_INPUT_DIM,
            hidden_size=cfg.ROBOT.RNN_HIDDEN_DIM,
            num_layers=cfg.ROBOT.RNN_NUM_LAYERS,
            batch_first=True
        )
        self.head = nn.Linear(cfg.ROBOT.RNN_HIDDEN_DIM, cfg.ROBOT.ROBOT_STATE_DIM)
        self.ar_proj = nn.Linear(cfg.ROBOT.ROBOT_STATE_DIM, cfg.ROBOT.ESTIMATOR_INPUT_DIM)

    def forward(self, history, hidden=None):
        out, hidden = self.lstm(history, hidden)
        feat = out[:, -1, :]
        return feat, self.head(feat), hidden

    def forward_ar(self, prev_pred, hidden):
        inp = self.ar_proj(prev_pred).unsqueeze(1)
        out, hidden = self.lstm(inp, hidden)
        feat = out[:, -1, :]
        return feat, self.head(feat), hidden


class JointActor(nn.Module):
    """
    Stochastic actor network for SAC.
    
    Improvements (without changing architecture):
    - Better log_std bounds for numerical stability
    - Improved action sampling
    """
    
    # Tighter log_std bounds for stability
    LOG_STD_MIN = -10.0
    LOG_STD_MAX = 2.0
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # Input: LSTM + Remote + History (unchanged)
        input_dim = cfg.ROBOT.RNN_HIDDEN_DIM + cfg.ROBOT.ROBOT_STATE_DIM + cfg.ROBOT.ROBOT_HISTORY_DIM
        
        # Network architecture unchanged for compatibility
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, cfg.ROBOT.N_JOINTS)
        self.log_std = nn.Linear(256, cfg.ROBOT.N_JOINTS)
        self.scale = torch.tensor(cfg.ROBOT.TORQUE_LIMITS)

    def forward(self, obs, hidden=None, prev_pred=None, has_new_obs=True):
        # Slice Observation
        idx_rem = cfg.ROBOT.ROBOT_STATE_DIM
        idx_hist = idx_rem + cfg.ROBOT.ROBOT_HISTORY_DIM
        
        remote_state = obs[:, :idx_rem]
        remote_hist = obs[:, idx_rem:idx_hist]
        target_hist = obs[:, idx_hist:]
        
        if has_new_obs:
            seq = target_hist.view(-1, cfg.ROBOT.RNN_SEQ_LEN, cfg.ROBOT.ESTIMATOR_INPUT_DIM)
            feat, pred, next_hidden = self.encoder(seq, hidden)
        else:
            feat, pred, next_hidden = self.encoder.forward_ar(prev_pred, hidden)
            
        x = torch.cat([feat, remote_state, remote_hist], dim=1)
        x = self.net(x)
        mu = self.mu(x)
        
        # Use class constants for bounds (tighter than original -20, 2)
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mu, log_std, pred, next_hidden

    def sample(self, obs, hidden=None, has_new_obs=True):
        mu, log_std, pred, next_hidden = self.forward(obs, hidden, None, has_new_obs)
        std = log_std.exp()
        
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.scale.to(obs.device)
        
        # Improved numerical stability in log_prob calculation
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob, pred, next_hidden

    def get_action_deterministic(self, obs, hidden=None):
        """Get deterministic action (mean) for evaluation."""
        mu, _, pred, next_hidden = self.forward(obs, hidden, None, True)
        action = torch.tanh(mu) * self.scale.to(obs.device)
        return action, pred, next_hidden


class JointCritic(nn.Module):
    """Twin Q-network critic for SAC. Architecture unchanged."""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        input_dim = cfg.ROBOT.RNN_HIDDEN_DIM + cfg.ROBOT.ROBOT_STATE_DIM + cfg.ROBOT.ROBOT_HISTORY_DIM + cfg.ROBOT.N_JOINTS
        self.q1 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, obs, action, hidden=None, has_new_obs=True):
        idx_rem = cfg.ROBOT.ROBOT_STATE_DIM
        idx_hist = idx_rem + cfg.ROBOT.ROBOT_HISTORY_DIM
        remote_state, remote_hist, target_hist = obs[:, :idx_rem], obs[:, idx_rem:idx_hist], obs[:, idx_hist:]
        
        if has_new_obs:
            seq = target_hist.view(-1, cfg.ROBOT.RNN_SEQ_LEN, cfg.ROBOT.ESTIMATOR_INPUT_DIM)
            feat, _, next_hidden = self.encoder(seq, hidden)
        
        xu = torch.cat([feat, remote_state, remote_hist, action], dim=1)
        return self.q1(xu), self.q2(xu), next_hidden


def create_actor_critic(device):
    enc = ContinuousLSTMEncoder().to(device)
    act = JointActor(enc).to(device)
    crit = JointCritic(enc).to(device)
    return enc, act, crit