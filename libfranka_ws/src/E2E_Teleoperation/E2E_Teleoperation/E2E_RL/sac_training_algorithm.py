"""
E2E_Teleoperation/E2E_RL/sac_training_algorithm.py
"""

import numpy as np
import torch

class RecoveryBuffer:
    """
    Professional Replay Buffer.
    Stores:
    1. Standard RL transitions (obs, action, reward, next_obs, done)
    2. Teacher Actions (for Behavioral Cloning)
    3. True State Vectors (for Encoder Training)
    """
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # 1. Standard RL Buffers
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # 2. Specialized Buffers
        self.teacher_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.true_states = np.zeros((capacity, 14), dtype=np.float32) # 7pos + 7vel
        self.is_recovery = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, teacher_action, true_state, is_recovery):
        # Insert data at current pointer
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        
        self.teacher_actions[self.ptr] = teacher_action
        self.true_states[self.ptr] = true_state
        self.is_recovery[self.ptr] = float(is_recovery)
        
        # Advance pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_with_recovery_weight(self, batch_size, recovery_weight=2.0):
        """
        Samples a batch. Prioritizes 'Recovery' steps (where robot was failing)
        if recovery_weight > 1.0.
        """
        # Calculate sampling weights
        weights = np.ones(self.size)
        # Identify recovery indices (where is_recovery is True)
        recovery_mask = self.is_recovery[:self.size, 0] > 0.5
        weights[recovery_mask] = recovery_weight
        
        # Normalize to probability distribution
        p_weights = weights / weights.sum()
        
        # Sample indices
        idxs = np.random.choice(self.size, size=batch_size, p=p_weights)
        
        # Return Tensors on GPU
        return {
            'obs': torch.FloatTensor(self.obs[idxs]).to(self.device),
            'next_obs': torch.FloatTensor(self.next_obs[idxs]).to(self.device),
            'actions': torch.FloatTensor(self.actions[idxs]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[idxs]).to(self.device),
            'dones': torch.FloatTensor(self.dones[idxs]).to(self.device),
            'teacher_actions': torch.FloatTensor(self.teacher_actions[idxs]).to(self.device),
            'true_state_vector': torch.FloatTensor(self.true_states[idxs]).to(self.device)
        }
        
    @property
    def current_size(self):
        return self.size