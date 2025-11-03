"""
Rollout buffer is a data container that stores the explored experience during interaction with environment.

It will be used for calculating advantages and returns, and feeding data to the policy update.

Store data:
- delayed_sequences
- remote_states
- actions, log_probs, values
- rewards, dones
- predicted_targets (from policy)
- true_targets (from environment)
"""

import numpy as np
import torch
from typing import Dict, List, Optional

class RolloutBuffer:
 
    def __init__(self, buffer_size: int):    
        
        self.buffer_size = buffer_size # The agent will collect this many steps for each policy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize lists to store trajectory data
        self.delayed_sequences: List[np.ndarray] = []
        self.remote_states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = [] # Tracks episode terminations
        self.predicted_targets: List[np.ndarray] = []
        self.true_targets: List[np.ndarray] = []

        # Initialize pointers
        self.ptr: int = 0          # Current position in the buffer
        self.path_start_idx: int = 0 # Index of the start of the current trajectory (not strictly needed for basic PPO)
        self.full: bool = False    # Flag indicating if buffer has wrapped around

    def reset(self):
        """Clear all data from the buffer."""
        self.delayed_sequences.clear()
        self.remote_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.predicted_targets.clear()
        self.true_targets.clear()
        self.ptr = 0
        self.full = False
        self.path_start_idx = 0

    def add(
        self,
        delayed_sequence: np.ndarray,
        remote_state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        predicted_target: np.ndarray,
        true_target: np.ndarray
    ):
        """Add new experience to the buffer (circular append when full)."""
        
        if len(self.rewards) < self.buffer_size:
            # Append until buffer fills
            self.delayed_sequences.append(delayed_sequence)
            self.remote_states.append(remote_state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)
            self.predicted_targets.append(predicted_target)
            self.true_targets.append(true_target)
        elif len(self.rewards) == self.buffer_size:
            # Overwrite oldest data (circular buffer)
            self.full = True
            idx = self.ptr % self.buffer_size
            self.delayed_sequences[idx] = delayed_sequence
            self.remote_states[idx] = remote_state
            self.actions[idx] = action
            self.log_probs[idx] = log_prob
            self.values[idx] = value
            self.rewards[idx] = reward
            self.dones[idx] = done
            self.predicted_targets[idx] = predicted_target
            self.true_targets[idx] = true_target
        else:
            raise RuntimeError(
                f"Buffer state inconsistency: len(rewards)={len(self.rewards)} > "
                f"buffer_size={self.buffer_size}. This indicates external corruption."
            )

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute advantages and returns using GAE."""
        """
        GAE: Generalized Advantage Estimation
        A_t = \delta_t + (\gamma * \lambda) * \delta_{t+1} + (\gamma * \lambda)^2 * \delta_{t+2} + ...

        where \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t) is the temporal difference error.
        """
        # Convert lists to numpy arrays for calculation
        values_np = np.array(self.values + [last_value]) # Append last value for bootstrap
        rewards_np = np.array(self.rewards)
        dones_np = np.array(self.dones)

        advantages = np.zeros_like(rewards_np)
        last_gae_lam = 0
        n_steps = len(rewards_np)

        # The new coming data (at T) is stored at the end of the list
        # In order to compute the GAE at timestep t, t+1, ...., T, we have to iterate backwards
        for t in reversed(range(n_steps)):
            next_non_terminal = 1.0 - dones_np[t]
            next_value = values_np[t + 1]

            # Calculate the TD error (delta) for a single timestep t.
            # delta_t = r_t + γ * V(s_{t+1}) - V(s_t)
            # V(s_t) is the value function estimate for the current state s_t.
            # The term (r_t + γ * V(s_{t+1})) is the TD target. It's an one more step ahead estimation
            # of the value of state s_t, using the real reward r_t and the estimated value of the next state.
            # If the episode terminates at step t, the value of the next state V(s_{t+1}) is 0.
            delta = rewards_np[t] + gamma * next_value * next_non_terminal - values_np[t]

            # Calculate GAE
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        # Calculate returns (target for value function)
        returns = advantages + values_np[:-1] # V_target = A + V(s_t)

        return advantages, returns

    def get_prediction_data(self) -> Dict[str, torch.Tensor]:
        """
        Extract prediction data WITHOUT clearing buffer.
        
        Critical for Phase 1: enables supervised LSTM loss computation
        while keeping buffer intact for get_policy_data().
        
        Returns:
            Dictionary with 'predicted_targets' and 'true_targets' tensors.
        """
        if len(self.predicted_targets) == 0:
            raise ValueError("No prediction data in buffer.")
        
        data = {
            'predicted_targets': torch.FloatTensor(
                np.array(self.predicted_targets)
            ).to(self.device),
            'true_targets': torch.FloatTensor(
                np.array(self.true_targets)
            ).to(self.device),
            'num_samples': len(self.predicted_targets)
        }
        return data
    
    def get_policy_data(self, advantages: np.ndarray, returns: np.ndarray) -> Dict[str, torch.Tensor]:
        """Loading data for training and clear the buffer afterwards."""
        if len(self.rewards) == 0:
            raise ValueError("Buffer is empty, cannot get data.")
        
        assert len(self.rewards) == len(advantages) == len(returns), "Data length mismatch!"

        # Normalize advantages (good practice for PPO)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        data = {
            'delayed_sequences': torch.FloatTensor(np.array(self.delayed_sequences)).to(self.device),
            'remote_states': torch.FloatTensor(np.array(self.remote_states)).to(self.device),
            'actions': torch.FloatTensor(np.array(self.actions)).to(self.device),
            'old_log_probs': torch.FloatTensor(self.log_probs).to(self.device),
            'old_values': torch.FloatTensor(self.values).to(self.device),
            'advantages': torch.FloatTensor(advantages).to(self.device),
            'returns': torch.FloatTensor(returns).to(self.device),
            'predicted_targets': torch.FloatTensor(np.array(self.predicted_targets)).to(self.device),
            'true_targets': torch.FloatTensor(np.array(self.true_targets)).to(self.device)
        }
        
        # Clear buffer after extracting data
        self.reset()
        return data

    def log_gradient_norms(self, module) -> Dict[str, float]:
        """Utility method to extract LSTM parameter gradient magnitudes for debugging."""
        grad_norms = {}
        for name, param in module.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
            else:
                grad_norms[name] = 0.0
        return grad_norms
    
    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        # Use the actual length of one of the stored lists
        return len(self.rewards)