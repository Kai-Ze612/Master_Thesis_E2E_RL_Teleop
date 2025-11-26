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

# Constants matching your config
N_JOINTS = 7
# Input to LSTM: (q, qd, delay_scalar) -> 7 + 7 + 1 = 15
LSTM_INPUT_DIM = 2 * N_JOINTS + 1 
HIDDEN_DIM = 1024
SEQ_LEN = 20  # RNN_SEQUENCE_LENGTH

class StateEstimator(nn.Module):
    """
    LSTM-based State Estimator for Autoregressive Prediction.
    Input: Sequence of [q, qd, delay_scalar] (Dim 15)
    Output: Residual prediction for ONE step ahead [delta_q, delta_qd] (Dim 14)
    """
    def __init__(self, input_dim=LSTM_INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=2):
        super(StateEstimator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Layer Norm on input for stability
        self.input_ln = nn.LayerNorm(input_dim)
        
        # LSTM Core
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Prediction Head (Predicts 14-dim residual)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Mish(),
            nn.Linear(256, 2 * N_JOINTS) # Output: 14 (q + qd)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 15)
        x = self.input_ln(x)
        
        # LSTM forward
        out, _ = self.lstm(x)
        
        # Take the last time step output
        last_hidden = out[:, -1, :]
        
        # Predict residual
        residual = self.prediction_head(last_hidden)
        
        return residual, last_hidden

class Actor(nn.Module):
    """
    SAC Actor Network.
    """
    def __init__(self, state_dim, action_dim=N_JOINTS, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            return torch.tanh(mean), mean, log_std
            
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Enforce action bounds if necessary (usually handled by tanh scaling outside)
        return action, normal.log_prob(x_t), x_t