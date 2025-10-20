"""
Custom SAC policy with learned trajectory predictor.

This policy has two components:
1. Predictor Network: Learns to predict current target position from delayed observations
2. Controller Network: Uses predicted target to compute control actions

Both are trained end-to-end with RL to optimize tracking performance under delay.
"""

import logging
import sys
import os
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import List

from Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    TARGET_HISTORY_LEN,
    ACTION_HISTORY_LEN
)

logger = logging.getLogger(__name__)

class PredictorControllerExtractor(BaseFeaturesExtractor):
    
    def __init__(
        self, 
        observation_space: spaces.Box, 
        features_dim: int = 256,
        predictor_arch: list = [128, 128, 64],
        controller_arch: list = [512],
    ):
        super().__init__(observation_space, features_dim)
        
        q_dim = N_JOINTS
        qd_dim = N_JOINTS
        delayed_q_dim = N_JOINTS
        q_hist_dim = N_JOINTS * TARGET_HISTORY_LEN
        qd_hist_dim = N_JOINTS * TARGET_HISTORY_LEN
        delay_dim = 1
        action_hist_dim = N_JOINTS * ACTION_HISTORY_LEN
        
        start = 0
        self.q_slice = slice(start, start + q_dim); start += q_dim
        self.qd_slice = slice(start, start + qd_dim); start += qd_dim
        self.delayed_q_slice = slice(start, start + delayed_q_dim); start += delayed_q_dim
        self.q_hist_slice = slice(start, start + q_hist_dim); start += q_hist_dim
        self.qd_hist_slice = slice(start, start + qd_hist_dim); start += qd_hist_dim
        self.delay_slice = slice(start, start + delay_dim); start += delay_dim
        self.action_hist_slice = slice(start, start + action_hist_dim)
        
        predictor_input_dim = delayed_q_dim + q_hist_dim + qd_hist_dim + delay_dim
        robot_state_dim = q_dim + qd_dim + action_hist_dim
        controller_input_dim = robot_state_dim + N_JOINTS
        
        predictor_layers = []
        in_dim = predictor_input_dim
        for hidden_dim in predictor_arch:
            predictor_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        predictor_layers.append(nn.Linear(in_dim, N_JOINTS))
        self.predictor = nn.Sequential(*predictor_layers)
        
        controller_layers = []
        in_dim = controller_input_dim
        for hidden_dim in controller_arch:
            controller_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        controller_layers.append(nn.Linear(in_dim, features_dim))
        self.controller_features = nn.Sequential(*controller_layers)

        logger.info("Initialized PredictorControllerExtractor policy.")
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        remote_q = observations[:, self.q_slice]
        remote_qd = observations[:, self.qd_slice]
        delayed_target_q = observations[:, self.delayed_q_slice]
        target_q_history = observations[:, self.q_hist_slice]
        target_qd_history = observations[:, self.qd_hist_slice]
        delay_magnitude = observations[:, self.delay_slice]
        action_history = observations[:, self.action_hist_slice]

        predictor_input = torch.cat([
            delayed_target_q, target_q_history, target_qd_history, delay_magnitude
        ], dim=1)
        predicted_current_q = self.predictor(predictor_input)

        robot_state = torch.cat([remote_q, remote_qd, action_history], dim=1)
        controller_input = torch.cat([robot_state, predicted_current_q], dim=1)
        features = self.controller_features(controller_input)

        return features
    
class LinearInterpolationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        q_dim = N_JOINTS
        qd_dim = N_JOINTS
        delayed_q_dim = N_JOINTS
        q_hist_dim = N_JOINTS * TARGET_HISTORY_LEN
        qd_hist_dim = N_JOINTS * TARGET_HISTORY_LEN
        delay_dim = 1
        action_hist_dim = N_JOINTS * ACTION_HISTORY_LEN

        start = 0
        self.q_slice = slice(start, start + q_dim); start += q_dim
        self.qd_slice = slice(start, start + qd_dim); start += qd_dim
        self.delayed_q_slice = slice(start, start + delayed_q_dim); start += delayed_q_dim
        self.q_hist_slice = slice(start, start + q_hist_dim); start += q_hist_dim
        self.qd_hist_slice = slice(start, start + qd_hist_dim); start += qd_hist_dim
        self.delay_slice = slice(start, start + delay_dim); start += delay_dim
        self.action_hist_slice = slice(start, start + action_hist_dim)
        
        robot_state_dim = q_dim + qd_dim + action_hist_dim
        controller_input_dim = robot_state_dim + N_JOINTS
        
        self.feature_projection = nn.Linear(controller_input_dim, features_dim)
        
        logger.info("Initialized SIMPLIFIED LinearInterpolationExtractor policy (single linear layer).")
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        remote_q = observations[:, self.q_slice]
        remote_qd = observations[:, self.qd_slice]
        delayed_target_q = observations[:, self.delayed_q_slice]
        target_qd_history = observations[:, self.qd_hist_slice]
        delay_magnitude = observations[:, self.delay_slice]
        action_history = observations[:, self.action_hist_slice]
        
        last_known_qd = target_qd_history[:, -N_JOINTS:]
        delay_seconds = delay_magnitude * (1.0 / 5.0)  # Heuristic for delay scaling
        predicted_current_q = delayed_target_q + last_known_qd * delay_seconds
        
        robot_state = torch.cat([remote_q, remote_qd, action_history], dim=1)
        controller_input = torch.cat([robot_state, predicted_current_q], dim=1)
        
        features = self.feature_projection(controller_input)

        return features
    
def get_policy_kwargs(policy_type: str, net_arch: List[int]) -> dict:
    """Returns the policy_kwargs dictionary for the specified policy type."""
    if policy_type == 'learned_predictor':
        return {
            "features_extractor_class": PredictorControllerExtractor,
            "features_extractor_kwargs": dict(features_dim=net_arch[0]),
            "net_arch": [],
        }
    elif policy_type == 'interpolation':
        return {
            "features_extractor_class": LinearInterpolationExtractor,
            "features_extractor_kwargs": dict(features_dim=net_arch[0]),
            "net_arch": [],
        }
    else: # 'baseline'
        return {"net_arch": net_arch}