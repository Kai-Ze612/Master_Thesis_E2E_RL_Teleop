"""
SAC Policy Network with Continuous LSTM Encoder

This is an E2E model where:
1. LSTM maintains hidden state across control steps
2. When observation arrives: process observation sequence
3. When no observation: use previous prediction as input (autoregressive)

This matches real-world deployment behavior where observations arrive sporadically
due to network delay, but control must output every 4ms (250Hz).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Normal

import E2E_Teleoperation.config.robot_config as cfg


class ContinuousLSTMEncoder(nn.Module):
    """
    LSTM encoder that supports both:
    1. Standard forward pass (when observation available)
    2. Autoregressive forward pass (when no observation)
    
    Hidden state is maintained externally and passed in each call.
    """
    def __init__(self):
        super().__init__()
        
        self.state_dim = cfg.ESTIMATOR_STATE_DIM   # 15 (q, qd, delay)
        self.output_dim = cfg.ESTIMATOR_OUTPUT_DIM  # 14 (q, qd)
        self.hidden_dim = cfg.RNN_HIDDEN_DIM        # 256
        self.num_layers = cfg.RNN_NUM_LAYERS        # 2
        
        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # Prediction head: LSTM hidden -> predicted state (14D)
        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )
        
        # Autoregressive input projection: predicted state (14D) -> LSTM input (15D)
        # This learns to convert a prediction back into LSTM input format
        self.ar_projection = nn.Sequential(
            nn.Linear(self.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.state_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights for stable training."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Forget gate bias = 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)
    
    def forward(
        self,
        history_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standard forward pass when observation sequence is available.
        
        Args:
            history_seq: (batch, seq_len, 15) delayed observation sequence
            hidden: (h, c) LSTM hidden state, or None to initialize
            
        Returns:
            lstm_features: (batch, hidden_dim) features for policy
            pred_state: (batch, 14) predicted current state
            next_hidden: updated (h, c)
        """
        batch_size = history_seq.shape[0]
        device = history_seq.device
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        
        # LSTM forward
        lstm_out, next_hidden = self.lstm(history_seq, hidden)
        
        # Last timestep features
        lstm_features = lstm_out[:, -1, :]
        
        # Predict state
        pred_state = self.pred_head(lstm_features)
        
        return lstm_features, pred_state, next_hidden
    
    def forward_ar(
        self,
        prev_prediction: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Autoregressive forward pass when no observation available.
        Uses previous prediction as input.
        
        Args:
            prev_prediction: (batch, 14) previous predicted state
            hidden: (h, c) LSTM hidden state (REQUIRED, not optional)
            
        Returns:
            lstm_features: (batch, hidden_dim) features for policy
            pred_state: (batch, 14) new predicted state
            next_hidden: updated (h, c)
        """
        # Project prediction to LSTM input format
        ar_input = self.ar_projection(prev_prediction)  # (batch, 15)
        
        # Add sequence dimension: (batch, 1, 15)
        ar_input = ar_input.unsqueeze(1)
        
        # LSTM step
        lstm_out, next_hidden = self.lstm(ar_input, hidden)
        
        # Features and prediction
        lstm_features = lstm_out[:, -1, :]
        pred_state = self.pred_head(lstm_features)
        
        return lstm_features, pred_state, next_hidden


class JointActor(nn.Module):
    """
    Actor network that uses ContinuousLSTMEncoder.
    
    Supports two modes:
    1. has_new_obs=True: Use observation sequence
    2. has_new_obs=False: Use autoregressive prediction
    """
    def __init__(
        self,
        shared_encoder: ContinuousLSTMEncoder,
        action_dim: int = cfg.N_JOINTS,
        hidden_dims: list = cfg.MLP_HIDDEN_DIMS,
    ):
        super().__init__()
        
        self.encoder = shared_encoder
        
        # Dimension calculations
        self.robot_state_dim = cfg.ROBOT_STATE_DIM       # 14
        self.target_hist_dim = cfg.TARGET_HISTORY_DIM   # 150 (10 * 15)
        self.robot_hist_dim = cfg.ROBOT_HISTORY_DIM     # 140 (10 * 14)
        
        # Policy input: LSTM features + robot state + robot history
        policy_input_dim = cfg.RNN_HIDDEN_DIM + self.robot_state_dim + self.robot_hist_dim
        
        # Policy backbone
        layers = []
        last_dim = policy_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        
        # Output heads
        self.fc_mean = nn.Linear(last_dim, action_dim)
        self.fc_log_std = nn.Linear(last_dim, action_dim)
        
        # Initialize small weights
        nn.init.uniform_(self.fc_mean.weight, -1e-3, 1e-3)
        nn.init.constant_(self.fc_mean.bias, 0.0)
        
        self.register_buffer('action_scale', torch.tensor(cfg.MAX_ACTION_TORQUE))
        self.register_buffer('action_bias', torch.tensor(0.0))
    
    def forward(
        self,
        obs_flat: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prev_prediction: Optional[torch.Tensor] = None,
        has_new_obs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with support for both observation and AR modes.
        
        Args:
            obs_flat: (batch, obs_dim) flattened observation
            hidden: LSTM hidden state
            prev_prediction: (batch, 14) previous prediction (for AR mode)
            has_new_obs: If True, use observation. If False, use AR.
            
        Returns:
            mean: (batch, action_dim) action mean
            log_std: (batch, action_dim) action log std
            pred_state: (batch, 14) predicted state
            next_hidden: updated LSTM hidden state
        """
        batch_size = obs_flat.shape[0]
        device = obs_flat.device
        
        # Extract components from observation
        remote_state = obs_flat[:, :self.robot_state_dim]
        robot_history = obs_flat[:, self.robot_state_dim:-self.target_hist_dim]
        target_history = obs_flat[:, -self.target_hist_dim:]
        
        # Encoder pass (observation or AR mode)
        if has_new_obs:
            # Reshape target history to sequence
            history_seq = target_history.view(batch_size, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
            lstm_features, pred_state, next_hidden = self.encoder(history_seq, hidden)
        else:
            # Autoregressive mode
            if prev_prediction is None:
                raise ValueError("prev_prediction required when has_new_obs=False")
            if hidden is None:
                raise ValueError("hidden state required when has_new_obs=False")
            lstm_features, pred_state, next_hidden = self.encoder.forward_ar(prev_prediction, hidden)
        
        # Policy forward (detach LSTM features to not backprop policy loss into encoder)
        policy_input = torch.cat([lstm_features.detach(), remote_state, robot_history], dim=1)
        x = self.backbone(policy_input)
        
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, cfg.LOG_STD_MIN, cfg.LOG_STD_MAX)
        
        return mean, log_std, pred_state, next_hidden
    
    def sample(
        self,
        obs_flat: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prev_prediction: Optional[torch.Tensor] = None,
        has_new_obs: bool = True,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample action from policy.
        
        Returns:
            scaled_action: (batch, action_dim) action in Nm
            log_prob: (batch, 1) log probability
            raw_action: (batch, action_dim) pre-tanh action
            pred_state: (batch, 14) predicted state
            next_hidden: updated LSTM hidden state
        """
        mean, log_std, pred_state, next_hidden = self.forward(
            obs_flat, hidden, prev_prediction, has_new_obs
        )
        
        std = log_std.exp()
        normal_dist = Normal(mean, std)
        
        if deterministic:
            raw_action = mean
        else:
            raw_action = normal_dist.rsample()
        
        tanh_action = torch.tanh(raw_action)
        
        # Log prob with tanh correction
        log_prob = normal_dist.log_prob(raw_action)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        scaled_action = self.action_scale * tanh_action + self.action_bias
        
        return scaled_action, log_prob, raw_action, pred_state, next_hidden


class JointCritic(nn.Module):
    """
    Critic network that uses ContinuousLSTMEncoder.
    Twin Q-networks for SAC.
    """
    def __init__(
        self,
        shared_encoder: ContinuousLSTMEncoder,
        action_dim: int = cfg.N_JOINTS,
        hidden_dims: list = cfg.MLP_HIDDEN_DIMS
    ):
        super().__init__()
        
        self.encoder = shared_encoder
        
        self.robot_state_dim = cfg.ROBOT_STATE_DIM
        self.target_hist_dim = cfg.TARGET_HISTORY_DIM
        self.robot_hist_dim = cfg.ROBOT_HISTORY_DIM
        
        # Input: LSTM features + robot state + robot history + action
        input_dim = cfg.RNN_HIDDEN_DIM + self.robot_state_dim + self.robot_hist_dim + action_dim
        
        # Twin Q-networks
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
    
    def forward(
        self,
        obs_flat: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prev_prediction: Optional[torch.Tensor] = None,
        has_new_obs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute Q-values.
        
        Returns:
            q1: (batch, 1) Q-value from network 1
            q2: (batch, 1) Q-value from network 2
            next_hidden: updated LSTM hidden state
        """
        batch_size = obs_flat.shape[0]
        device = obs_flat.device
        
        # Extract components
        remote_state = obs_flat[:, :self.robot_state_dim]
        robot_history = obs_flat[:, self.robot_state_dim:-self.target_hist_dim]
        target_history = obs_flat[:, -self.target_hist_dim:]
        
        # Encoder pass
        if has_new_obs:
            history_seq = target_history.view(batch_size, cfg.RNN_SEQUENCE_LENGTH, cfg.ESTIMATOR_STATE_DIM)
            lstm_features, _, next_hidden = self.encoder(history_seq, hidden)
        else:
            if prev_prediction is None or hidden is None:
                raise ValueError("prev_prediction and hidden required for AR mode")
            lstm_features, _, next_hidden = self.encoder.forward_ar(prev_prediction, hidden)
        
        # Normalize action
        action_normalized = action / self.action_scale
        
        # Concatenate inputs
        xu = torch.cat([lstm_features, remote_state, robot_history, action_normalized], dim=1)
        
        return self.q1_net(xu), self.q2_net(xu), next_hidden


def create_actor_critic(device: str = 'cuda'):
    """Factory function to create actor-critic networks with shared encoder."""
    shared_encoder = ContinuousLSTMEncoder().to(device)
    actor = JointActor(shared_encoder, action_dim=cfg.N_JOINTS).to(device)
    critic = JointCritic(shared_encoder, action_dim=cfg.N_JOINTS).to(device)
    return shared_encoder, actor, critic