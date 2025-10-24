"""
Rollout buffer for storing and managing PPO training data,
including specific data needed for the recurrent predictor.
"""

import numpy as np
import torch
from typing import Dict, List, Optional

class RolloutBuffer:
    """
    Buffer for storing trajectories experienced by a PPO agent interacting
    with the environment. Used for calculating advantages and returns,
    and feeding data to the policy update.

    Stores data specifically required for the RecurrentPPOPolicy:
    - delayed_sequences
    - remote_states
    - actions, log_probs, values
    - rewards, dones
    - predicted_targets (from policy)
    - true_targets (from environment)
    """

    def __init__(self, buffer_size: int, device: torch.device = torch.device('cuda')):
        """
        Args:
            buffer_size: Max number of transitions to store. Corresponds to PPO_ROLLOUT_STEPS.
            device: PyTorch device (e.g., 'cuda' or 'cpu').
        """
        self.buffer_size = buffer_size
        self.device = device

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
        self.full = False # Should be reset as well
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
        """
        Add a single transition step to the buffer.

        Args:
            delayed_sequence: Delayed observation sequence (seq_len, features).
            remote_state: Current remote state (features,).
            action: Action taken by the policy (action_dim,).
            log_prob: Log probability of the action.
            value: Value estimate from the critic V(s).
            reward: Reward received after taking the action.
            done: Whether the episode terminated after this step.
            predicted_target: Target state predicted by the policy (target_dim,).
            true_target: Ground truth target state from the environment (target_dim,).
        """
        if len(self.rewards) < self.buffer_size:
            # Append new data if buffer is not full
            self.delayed_sequences.append(delayed_sequence)
            self.remote_states.append(remote_state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)
            self.predicted_targets.append(predicted_target)
            self.true_targets.append(true_target)
        else:
            # Overwrite oldest data if buffer is full (circular buffer behavior)
            # Although for standard PPO, we usually clear after getting data
            self.full = True
            idx = self.ptr
            self.delayed_sequences[idx] = delayed_sequence
            self.remote_states[idx] = remote_state
            self.actions[idx] = action
            self.log_probs[idx] = log_prob
            self.values[idx] = value
            self.rewards[idx] = reward
            self.dones[idx] = done
            self.predicted_targets[idx] = predicted_target
            self.true_targets[idx] = true_target

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
        """
        Retrieve all collected data as PyTorch tensors, along with calculated
        advantages and returns. Resets the buffer after data retrieval.

        Args:
            advantages: Calculated advantages from `compute_returns_and_advantages`.
            returns: Calculated returns from `compute_returns_and_advantages`.

        Returns:
            A dictionary containing all trajectory data as tensors on the specified device.
        """
        if not self.full and self.ptr == 0:
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
        return self.buffer_size if self.full else self.ptr