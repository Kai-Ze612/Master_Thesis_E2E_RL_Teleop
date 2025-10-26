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
        """Add new experience to the buffer."""
        
        # Append new data if buffer is not full
        if len(self.rewards) < self.buffer_size:
            self.delayed_sequences.append(delayed_sequence)
            self.remote_states.append(remote_state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)
            self.predicted_targets.append(predicted_target)
            self.true_targets.append(true_target)

        # Increment pointer and wrap around if necessary
        self.ptr = (self.ptr + 1) % self.buffer_size


    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        Call this method after the rollout is complete and before `get()`.

        Args:
            last_value: Value estimate V(s_T) for the last state in the rollout
                        (used for bootstrapping if the episode didn't end).
            gamma: Discount factor.
            gae_lambda: Lambda factor for GAE (e.g., 0.95).

        Returns:
            advantages: Calculated advantages for each step.
            returns: Calculated returns (targets for the value function).
        """
        # Convert lists to numpy arrays for calculation
        values_np = np.array(self.values + [last_value]) # Append last value for bootstrap
        rewards_np = np.array(self.rewards)
        dones_np = np.array(self.dones)

        advantages = np.zeros_like(rewards_np)
        last_gae_lam = 0
        n_steps = len(rewards_np)

        for t in reversed(range(n_steps)):
            # If the episode ended at step t, the value of the next state V(s_{t+1}) is 0
            # Otherwise, use the estimated value V(s_{t+1})
            # Handle the very last step: if dones_np[t] is True, next_non_terminal is 0
            next_non_terminal = 1.0 - dones_np[t]
            next_value = values_np[t + 1] # V(s_{t+1})

            # Calculate TD error (delta)
            delta = rewards_np[t] + gamma * next_value * next_non_terminal - values_np[t]

            # Calculate GAE
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        # Calculate returns (target for value function)
        returns = advantages + values_np[:-1] # V_target = A + V(s_t)

        return advantages, returns


    def get(self, advantages: np.ndarray, returns: np.ndarray) -> Dict[str, torch.Tensor]:
        """ Advantage computes a scalar score for each"""
        if len(self.rewards) == 0:
            raise ValueError("Buffer is empty, cannot get data.")

        # Ensure data consistency
        assert len(self.rewards) == len(advantages) == len(returns), "Data length mismatch!"

        # --- Normalize Advantages --- (Common practice in PPO)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Convert all stored lists to tensors
        data = {
            # Observations for the policy
            'delayed_sequences': torch.FloatTensor(np.array(self.delayed_sequences)).to(self.device),
            'remote_states': torch.FloatTensor(np.array(self.remote_states)).to(self.device),
            # Actions taken and their log probs (from the policy *during rollout*)
            'actions': torch.FloatTensor(np.array(self.actions)).to(self.device),
            'old_log_probs': torch.FloatTensor(self.log_probs).to(self.device),
            # Value estimates (from the policy *during rollout*)
            'old_values': torch.FloatTensor(self.values).to(self.device),
            # Targets for training
            'advantages': torch.FloatTensor(advantages).to(self.device),
            'returns': torch.FloatTensor(returns).to(self.device),
            # Data for prediction loss
            'predicted_targets': torch.FloatTensor(np.array(self.predicted_targets)).to(self.device),
            'true_targets': torch.FloatTensor(np.array(self.true_targets)).to(self.device)
        }

        # Clear the buffer after retrieving data (standard for on-policy PPO)
        self.reset()

        return data

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        # Use the actual length of one of the stored lists
        return len(self.rewards)