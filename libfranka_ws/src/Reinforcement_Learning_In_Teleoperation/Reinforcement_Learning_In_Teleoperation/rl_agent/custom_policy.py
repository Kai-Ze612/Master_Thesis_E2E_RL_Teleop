"""
Custom SAC policy with learned trajectory predictor.

This policy has two components:
1. Predictor Network: Learns to predict current target position from delayed observations
2. Controller Network: Uses predicted target to compute control actions

Both are trained end-to-end with RL to optimize tracking performance.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PredictorControllerExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with explicit learned trajectory predictor.
    
    Architecture:
        Input: [joint_pos(7), joint_vel(7), gravity(7), delayed_target(3),
                target_history(30), delay_magnitude(1), action_history(35)]
        
        Predictor Branch:
            Input: delayed_target(3) + target_history(30) + delay(1) = 34
            Output: predicted_current_target(3)
        
        Controller Branch:
            Input: robot_state(56) + predicted_target(3) = 59
            Output: features(256)
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Observation structure (total 90):
        # [0:7]     joint_pos (7)
        # [7:14]    joint_vel (7)
        # [14:21]   gravity (7)
        # [21:24]   delayed_target (3)
        # [24:54]   target_history (10 × 3 = 30)
        # [54:55]   delay_magnitude (1)
        # [55:90]   action_history (7 × 5 = 35)
        
        self.predictor_input_dim = 34  # delayed(3) + history(30) + delay(1)
        self.robot_state_dim = 56      # pos(7) + vel(7) + gravity(7) + actions(35)
        
        print(f"\n{'='*70}")
        print(f"CUSTOM POLICY: Predictor + Controller")
        print(f"{'='*70}")
        print(f"Predictor input dim: {self.predictor_input_dim}")
        print(f"Robot state dim: {self.robot_state_dim}")
        print(f"Features output dim: {features_dim}")
        print(f"{'='*70}\n")
        
        # ═══════════════════════════════════════════════════════════
        # PREDICTOR NETWORK - Learns to predict current target
        # ═══════════════════════════════════════════════════════════
        self.predictor = nn.Sequential(
            nn.Linear(self.predictor_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output: predicted current position (x, y, z)
        )
        
        # ═══════════════════════════════════════════════════════════
        # CONTROLLER NETWORK - Uses prediction for control
        # ═══════════════════════════════════════════════════════════
        self.controller_features = nn.Sequential(
            nn.Linear(self.robot_state_dim + 3, 512),  # robot_state + predicted_target
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        # For logging/debugging
        self.last_predicted_target = None
        self.last_delayed_target = None
        self.prediction_offset = 0.0
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Predict current target, then compute control features.
        
        Args:
            observations: [batch_size, 90]
        
        Returns:
            features: [batch_size, features_dim]
        """
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Parse observation into components
        # ═══════════════════════════════════════════════════════════
        joint_pos = observations[:, 0:7]
        joint_vel = observations[:, 7:14]
        gravity = observations[:, 14:21]
        delayed_target = observations[:, 21:24]
        target_history = observations[:, 24:54]
        delay_magnitude = observations[:, 54:55]
        action_history = observations[:, 55:90]
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: PREDICT CURRENT TARGET using learned predictor
        # ═══════════════════════════════════════════════════════════
        predictor_input = torch.cat([
            delayed_target,
            target_history,
            delay_magnitude
        ], dim=1)
        
        # Predictor learns: delayed + history + delay → current position
        predicted_current_target = self.predictor(predictor_input)
        
        # Store for visualization/debugging
        if observations.shape[0] == 1:  # Single observation (not batch)
            self.last_predicted_target = predicted_current_target.detach().cpu()
            self.last_delayed_target = delayed_target.detach().cpu()
            self.prediction_offset = torch.norm(
                predicted_current_target - delayed_target
            ).item()
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: COMPUTE CONTROL FEATURES with predicted target
        # ═══════════════════════════════════════════════════════════
        robot_state = torch.cat([
            joint_pos,
            joint_vel,
            gravity,
            action_history
        ], dim=1)
        
        controller_input = torch.cat([
            robot_state,
            predicted_current_target
        ], dim=1)
        
        # Controller learns: state + predicted_target → control features
        features = self.controller_features(controller_input)
        
        return features


def create_predictor_policy():
    """
    Create policy kwargs for SAC with learned predictor.
    
    Returns:
        dict: Policy configuration for SAC
    """
    policy_kwargs = {
        "features_extractor_class": PredictorControllerExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 256],  # Hidden layers for actor and critic
    }
    
    return policy_kwargs


if __name__ == "__main__":
    # Test the policy structure
    print("Testing PredictorControllerExtractor...")
    
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(90,), dtype='float32')
    extractor = PredictorControllerExtractor(obs_space, features_dim=256)
    
    # Test forward pass
    test_obs = torch.randn(4, 90)  # Batch of 4 observations
    features = extractor(test_obs)
    
    print(f"\nTest passed!")
    print(f"Input shape: {test_obs.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: [4, 256]")