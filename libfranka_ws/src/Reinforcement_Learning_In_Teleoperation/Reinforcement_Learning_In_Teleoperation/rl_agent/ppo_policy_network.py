"""
Custom recurrent PPO policy for end to end training.

This script defines the NN architecture, forward pass, action sampling, and action evaluation methods.

Architecture:
    1. RNN (LSTM): State prediction head (trained with supervised loss).
    2. Actor (MLP): Policy network head (trained with PPO actor loss).
    3. Critic (MLP): Value function head (trained with PPO critic loss).
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os
from typing import Tuple, Optional, Union

from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    MAX_TORQUE_COMPENSATION,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    RNN_SEQUENCE_LENGTH, # Should be same as STATE_BUFFER_LENGTH
    PPO_MLP_HIDDEN_DIMS,
    PPO_ACTIVATION,
)

HiddenStateType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class RecurrentPPOPolicy(nn.Module):
    def __init__(
        self,
        rnn_hidden_dim: int = RNN_HIDDEN_DIM,
        rnn_num_layers: int = RNN_NUM_LAYERS,
        mlp_hidden_dims: Optional[list] = None,
        activation: str = PPO_ACTIVATION
    ):
        super().__init__()
    
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.seq_length = RNN_SEQUENCE_LENGTH
        self.activation_fn = self._get_activation(activation)
        
        if mlp_hidden_dims is None:
            mlp_hidden_dims = PPO_MLP_HIDDEN_DIMS
            
        self.feature_dim = N_JOINTS * 2  # q and qd
        
        # LSTM for state prediction
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True # Expect input shape (batch, seq_len, features)
        )
        
        # Prediction head (outputs predicted current q and qd)
        self.prediction_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 128),
            self.activation_fn(),
            nn.Linear(128, N_JOINTS * 2) # Output: Predicted [q_target, qd_target] (14 dims)
        )
        
        # PPO MLP networks
        policy_input_dim = (N_JOINTS * 2) * 2
        
        # Build shared MLP backbone for actor and critic
        layers = []
        last_dim = policy_input_dim
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(self.activation_fn())
            last_dim = hidden_dim

        # If mlp_hidden_dims is empty, policy_backbone is just an identity layer
        self.policy_backbone = nn.Sequential(*layers) if layers else nn.Identity()

        # Actor head (outputs mean of the action distribution)
        self.actor_mean = nn.Sequential(
            nn.Linear(last_dim, 128),
            self.activation_fn(),
            nn.Linear(128, N_JOINTS) # Output: Mean of torque compensation dist (7 dims)
        )

        # Log standard deviation (learnable parameter, shared across batch)
        # Initialized near zero for small initial std deviation
        self.actor_log_std = nn.Parameter(torch.zeros(1, N_JOINTS) - 0.5)

        # Critic head (outputs state value V(s))
        self.critic = nn.Sequential(
            nn.Linear(last_dim, 128),
            self.activation_fn(),
            nn.Linear(128, 1) # Output: State value V(s) (scalar)
        )

        self._initialize_weights()
        
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Helper to get activation function module."""
        if activation_name == "relu":
            return nn.ReLU
        elif activation_name == "tanh":
            return nn.Tanh
        elif activation_name == "elu":
            return nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Orthogonal init gain sqrt(2) is common for ReLU
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                # LSTM weights are already well-initialized by PyTorch
                pass
            
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_hidden_state(self, batch_size: int, device: torch.device) -> HiddenStateType:
        """
        Initialize LSTM hidden state.
        
        Args:
            batch_size: Number of sequences in batch
            device: Device to create tensors on
            
        Returns:
            Tuple of (h_0, c_0) for LSTM hidden and cell states
        """
        h_0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)
        c_0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)
        return (h_0, c_0)

    def predict_state(self, delayed_sequence: torch.Tensor,
                      hidden_state: Optional[HiddenStateType] = None
                      ) -> Tuple[torch.Tensor, HiddenStateType]:
        """
        Predict current local robot state from delayed observation sequence.
        
        Args:
            delayed_sequence: Shape (batch, seq_len, 14) - delayed [q, qd] observations
            hidden_state: Optional LSTM hidden state tuple (h, c)
            
        Returns:
            predicted_target: Shape (batch, 14) - predicted current [q, qd]
            new_hidden_state: Updated LSTM hidden state tuple
        """
        
        # Ensure input sequence length matches expected
        if delayed_sequence.shape[1] != self.seq_length:
            print(f"Warning: Input sequence length {delayed_sequence.shape[1]} does not match expected {self.seq_length}.")

        # Forward through LSTM
        # If hidden_state is None, LSTM uses default zero state
        lstm_output, new_hidden_state = self.lstm(delayed_sequence, hidden_state)

        # Use the output from the *last* time step in the sequence
        last_lstm_output = lstm_output[:, -1, :] # Shape: (batch, rnn_hidden_dim)

        # Pass through the prediction head
        predicted_target = self.prediction_head(last_lstm_output)

        return predicted_target, new_hidden_state

    def forward(self, delayed_sequence: torch.Tensor, remote_state: torch.Tensor,
                hidden_state: Optional[HiddenStateType] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, HiddenStateType]:
        """
        Complete forward pass for training/evaluation.
        
        Args:
            delayed_sequence: Shape (batch, seq_len, 14) - delayed observations
            remote_state: Shape (batch, 14) - current remote robot [q, qd]
            hidden_state: Optional LSTM hidden state
            
        Returns:
            predicted_target: Shape (batch, 14) - predicted current local [q, qd]
            action_mean: Shape (batch, 7) - mean of torque compensation
            action_std: Shape (batch, 7) - std of torque compensation
            value: Shape (batch, 1) - state value estimate
            new_hidden_state: Updated LSTM hidden state
        """
        
        # LSTM predicts the current target state
        predicted_target, new_hidden_state = self.predict_state(delayed_sequence, hidden_state)

        # PPO policy using predicted target and current remote state
        policy_input = torch.cat([predicted_target, remote_state], dim=-1) # End-to-End

        # Pass through shared policy backbone
        policy_features = self.policy_backbone(policy_input)

        # Actor: Calculate action distribution parameters
        action_mean = self.actor_mean(policy_features)
        action_log_std = self.actor_log_std.expand_as(action_mean) # Ensure shape matches mean
        action_std = torch.exp(action_log_std)

        # Critic: Estimate state value
        value = self.critic(policy_features)

        return predicted_target, action_mean, action_std, value, new_hidden_state

    def get_action(self, delayed_sequence: torch.Tensor, remote_state: torch.Tensor,
                   hidden_state: Optional[HiddenStateType] = None, deterministic: bool = False
                  ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, HiddenStateType]:
        """
        Sample action from policy.
        
        Args:
            delayed_sequence: Shape (batch, seq_len, 14) - delayed observations
            remote_state: Shape (batch, 14) - current remote robot state
            hidden_state: Optional LSTM hidden state
            deterministic: If True, return mean action; if False, sample from distribution
            
        Returns:
            action: Shape (batch, 7) - torque compensation (clamped to limits)
            log_prob: Shape (batch,) - log probability of action (None if deterministic)
            value: Shape (batch, 1) - state value estimate
            predicted_target: Shape (batch, 14) - predicted local state
            new_hidden_state: Updated LSTM hidden state
        """

        predicted_target, action_mean, action_std, value, new_hidden_state = self.forward(
            delayed_sequence, remote_state, hidden_state
        )

        # Create action distribution
        action_dist = Normal(action_mean, action_std)

        if deterministic:
            # Return mean action during evaluation/deployment
            action = action_mean
            log_prob = None
        else:
            # Sample action during training for exploration
            action = action_dist.sample()
            # Calculate log probability of the sampled action
            log_prob = action_dist.log_prob(action).sum(dim=-1) # Sum log probs across action dim

        # Clamp action to torque compensation limits
        action = torch.clamp(
            action,
            -torch.tensor(MAX_TORQUE_COMPENSATION, device=action.device, dtype=action.dtype),
            torch.tensor(MAX_TORQUE_COMPENSATION, device=action.device, dtype=action.dtype)
        )

        return action, log_prob, value, predicted_target, new_hidden_state

    def evaluate_actions(self, delayed_sequence: torch.Tensor, remote_state: torch.Tensor,
                         actions: torch.Tensor, hidden_state: Optional[HiddenStateType] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, HiddenStateType]:
        """
        Evaluate actions for PPO update (compute log_prob, entropy, value).
        
        Args:
            delayed_sequence: Shape (batch, seq_len, 14) - delayed observations
            remote_state: Shape (batch, 14) - current remote robot state
            actions: Shape (batch, 7) - actions taken during rollout
            hidden_state: Optional LSTM hidden state
            
        Returns:
            log_prob: Shape (batch,) - log probability of given actions
            entropy: Shape (batch,) - entropy of action distribution
            value: Shape (batch, 1) - state value estimate
            predicted_target: Shape (batch, 14) - predicted local state
            new_hidden_state: Updated LSTM hidden state
        """
       
        predicted_target, action_mean, action_std, value, new_hidden_state = self.forward(
            delayed_sequence, remote_state, hidden_state
        )

        # Create distribution based on current policy parameters
        action_dist = Normal(action_mean, action_std)

        # Calculate log probability of the actions *actually taken* during rollout
        log_prob = action_dist.log_prob(actions).sum(dim=-1)

        # Calculate entropy of the current action distribution (for exploration bonus)
        entropy = action_dist.entropy().sum(dim=-1)

        return log_prob, entropy, value, predicted_target, new_hidden_state

    def save(self, path: str):
        """Save model checkpoint."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save state dict and architecture info
        torch.save({
            'state_dict': self.state_dict(),
            'rnn_hidden_dim': self.rnn_hidden_dim,
            'rnn_num_layers': self.rnn_num_layers,
            'seq_length': self.seq_length,
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """Load model checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location=device)

        # Recreate model with saved architecture
        model = cls(
            rnn_hidden_dim=checkpoint.get('rnn_hidden_dim', RNN_HIDDEN_DIM),
            rnn_num_layers=checkpoint.get('rnn_num_layers', RNN_NUM_LAYERS),
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"Model loaded from {path} to {device}")
        return model