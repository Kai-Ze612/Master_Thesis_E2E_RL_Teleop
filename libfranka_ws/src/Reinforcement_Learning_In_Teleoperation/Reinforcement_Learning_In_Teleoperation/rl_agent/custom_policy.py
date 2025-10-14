"""
Custom SAC policy with learned trajectory predictor.

This policy has two components:
1. Predictor Network: Learns to predict current target position from delayed observations
2. Controller Network: Uses predicted target to compute control actions

Both are trained end-to-end with RL to optimize tracking performance under delay.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple


class PredictorControllerExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with explicit learned trajectory predictor.
    
    Architecture:
        Observation Input: [joint_pos(7), joint_vel(7), gravity(7), delayed_target(3),
                           target_history_pos(30), target_history_vel(30), 
                           delay_magnitude(1), action_history(35)]
        Total: 7+7+7+3+30+30+1+35 = 120 dimensions
        
        Predictor Branch:
            Input: delayed_target(3) + target_history_pos(30) + 
                   target_history_vel(30) + delay(1) = 64
            Output: predicted_current_target(3)
        
        Controller Branch:
            Input: robot_state(56) + predicted_target(3) = 59
            Output: features(256)
    """
    
    def __init__(
        self, 
        observation_space: spaces.Box, 
        features_dim: int = 256,
        predictor_arch: list = [128, 128, 64],
        controller_arch: list = [512],
    ):
        """
        Initialize predictor-controller architecture.
        
        Args:
            observation_space: Observation space from environment
            features_dim: Output feature dimension
            predictor_arch: Hidden layer sizes for predictor network
            controller_arch: Hidden layer sizes before final features
        """
        super().__init__(observation_space, features_dim)
        
        # Parse observation structure based on environment definition
        # Observation structure (total depends on history lengths):
        # [0:7]       joint_pos (7)
        # [7:14]      joint_vel (7)
        # [14:21]     gravity (7)
        # [21:24]     delayed_target (3)
        # [24:54]     target_history_pos (10 × 3 = 30)
        # [54:84]     target_history_vel (10 × 3 = 30)
        # [84:85]     delay_magnitude (1)
        # [85:120]    action_history (7 × 5 = 35)
        
        obs_dim = observation_space.shape[0]
        
        # Calculate dimensions
        self.joint_state_dim = 21  # pos(7) + vel(7) + gravity(7)
        self.action_history_dim = 35  # 7 joints × 5 history
        self.robot_state_dim = self.joint_state_dim + self.action_history_dim  # 56
        
        # Predictor input: delayed_target(3) + history_pos(30) + history_vel(30) + delay(1)
        self.predictor_input_dim = 3 + 30 + 30 + 1  # 64
        
        # Controller input: robot_state(56) + predicted_target(3)
        self.controller_input_dim = self.robot_state_dim + 3  # 59
        
        print(f"\n{'='*70}")
        print(f"CUSTOM POLICY INITIALIZATION")
        print(f"{'='*70}")
        print(f"Observation space dimension: {obs_dim}")
        print(f"Predictor Network:")
        print(f"  Input dim: {self.predictor_input_dim}")
        print(f"  Hidden layers: {predictor_arch}")
        print(f"  Output: 3 (predicted current position)")
        print(f"Controller Network:")
        print(f"  Input dim: {self.controller_input_dim}")
        print(f"  Hidden layers: {controller_arch}")
        print(f"  Output features: {features_dim}")
        print(f"{'='*70}\n")
        
        # ═══════════════════════════════════════════════════════════
        # PREDICTOR NETWORK - Learns trajectory prediction
        # ═══════════════════════════════════════════════════════════
        predictor_layers = []
        in_dim = self.predictor_input_dim
        
        for hidden_dim in predictor_arch:
            predictor_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # Stabilize training
            ])
            in_dim = hidden_dim
        
        # Output layer: predict 3D position
        predictor_layers.append(nn.Linear(in_dim, 3))
        
        self.predictor = nn.Sequential(*predictor_layers)
        
        # ═══════════════════════════════════════════════════════════
        # CONTROLLER NETWORK - Uses prediction for control
        # ═══════════════════════════════════════════════════════════
        controller_layers = []
        in_dim = self.controller_input_dim
        
        for hidden_dim in controller_arch:
            controller_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim
        
        # Final feature layer
        controller_layers.extend([
            nn.Linear(in_dim, features_dim),
            nn.ReLU()
        ])
        
        self.controller_features = nn.Sequential(*controller_layers)
        
        # For monitoring/debugging
        self.register_buffer('last_predicted_target', torch.zeros(3))
        self.register_buffer('last_delayed_target', torch.zeros(3))
        self.register_buffer('prediction_error_norm', torch.tensor(0.0))
        self._step_count = 0
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Predict current target, then compute control features.
        
        Args:
            observations: [batch_size, obs_dim]
        
        Returns:
            features: [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Parse observation into components
        # ═══════════════════════════════════════════════════════════
        joint_pos = observations[:, 0:7]
        joint_vel = observations[:, 7:14]
        gravity = observations[:, 14:21]
        delayed_target = observations[:, 21:24]
        target_history_pos = observations[:, 24:54]
        target_history_vel = observations[:, 54:84]
        delay_magnitude = observations[:, 84:85]
        action_history = observations[:, 85:120]
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: PREDICT CURRENT TARGET using learned predictor
        # ═══════════════════════════════════════════════════════════
        predictor_input = torch.cat([
            delayed_target,
            target_history_pos,
            target_history_vel,
            delay_magnitude
        ], dim=1)
        
        # Neural network learns: f(delayed, history, delay_mag) → current_position
        predicted_current_target = self.predictor(predictor_input)
        
        # Store for monitoring (only for single observation, not batch)
        if batch_size == 1:
            self.last_predicted_target.copy_(predicted_current_target[0].detach())
            self.last_delayed_target.copy_(delayed_target[0].detach())
            self.prediction_error_norm.copy_(
                torch.norm(predicted_current_target[0] - delayed_target[0]).detach()
            )
            
            # Periodic logging
            self._step_count += 1
            if self._step_count % 1000 == 0:
                print(f"Predictor stats (step {self._step_count}): "
                      f"prediction_offset={self.prediction_error_norm.item():.4f}")
        
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
        
        # Controller network: state + predicted_target → control features
        features = self.controller_features(controller_input)
        
        return features
    
    def get_predictor_stats(self) -> dict:
        """
        Get statistics about predictor performance.
        
        Returns:
            dict: Predictor statistics for logging
        """
        return {
            'predicted_target': self.last_predicted_target.cpu().numpy(),
            'delayed_target': self.last_delayed_target.cpu().numpy(),
            'prediction_offset': self.prediction_error_norm.item(),
        }


def create_predictor_policy(
    features_dim: int = 256,
    predictor_arch: list = None,
    controller_arch: list = None,
    actor_arch: list = None,
    critic_arch: list = None,
):
    """
    Create policy kwargs for SAC with learned predictor.
    
    Args:
        features_dim: Feature extractor output dimension
        predictor_arch: Hidden layers for predictor network
        controller_arch: Hidden layers for controller feature extractor
        actor_arch: Hidden layers for actor network (after features)
        critic_arch: Hidden layers for critic network (after features)
    
    Returns:
        dict: Policy configuration for SAC
    """
    # Default architectures
    if predictor_arch is None:
        predictor_arch = [128, 128, 64]
    if controller_arch is None:
        controller_arch = [512]
    if actor_arch is None:
        actor_arch = [256, 256]
    if critic_arch is None:
        critic_arch = [256, 256]
    
    policy_kwargs = {
        "features_extractor_class": PredictorControllerExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "predictor_arch": predictor_arch,
            "controller_arch": controller_arch,
        },
        "net_arch": {
            "pi": actor_arch,   # Actor network
            "qf": critic_arch,  # Critic network
        },
    }
    
    print(f"\nPolicy Configuration:")
    print(f"  Feature extractor: PredictorControllerExtractor")
    print(f"  Features dim: {features_dim}")
    print(f"  Actor architecture: {actor_arch}")
    print(f"  Critic architecture: {critic_arch}")
    print()
    
    return policy_kwargs


# ============================================================
# Alternative: Simpler Linear Interpolation Policy
# ============================================================

class LinearInterpolationExtractor(BaseFeaturesExtractor):
    """
    Simple baseline: Linear interpolation for trajectory prediction.
    
    Uses velocity-based linear extrapolation:
        predicted = delayed + velocity * delay_magnitude
    
    This serves as a baseline comparison to the learned predictor.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.robot_state_dim = 56  # pos(7) + vel(7) + gravity(7) + actions(35)
        
        print(f"\n{'='*70}")
        print(f"BASELINE POLICY: Linear Interpolation")
        print(f"{'='*70}")
        print(f"Prediction: velocity-based linear extrapolation")
        print(f"Features output dim: {features_dim}")
        print(f"{'='*70}\n")
        
        # Controller network (no predictor, uses interpolation)
        self.controller_features = nn.Sequential(
            nn.Linear(self.robot_state_dim + 3, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Linear interpolation + control."""
        # Parse observations
        joint_pos = observations[:, 0:7]
        joint_vel = observations[:, 7:14]
        gravity = observations[:, 14:21]
        delayed_target = observations[:, 21:24]
        target_history_vel = observations[:, 54:84]  # Use velocity history
        delay_magnitude = observations[:, 84:85]
        action_history = observations[:, 85:120]
        
        # Linear interpolation: use most recent velocity
        most_recent_velocity = target_history_vel[:, -3:]  # Last 3 values
        
        # Predict: position = delayed_position + velocity * delay_time
        # delay_magnitude is normalized (0-1), scale to approximate seconds
        delay_seconds = delay_magnitude * 0.2  # Max ~200ms
        predicted_current_target = delayed_target + most_recent_velocity * delay_seconds * 10.0
        
        # Controller input
        robot_state = torch.cat([joint_pos, joint_vel, gravity, action_history], dim=1)
        controller_input = torch.cat([robot_state, predicted_current_target], dim=1)
        
        return self.controller_features(controller_input)


def create_interpolation_policy(features_dim: int = 256):
    """Create policy with linear interpolation baseline."""
    policy_kwargs = {
        "features_extractor_class": LinearInterpolationExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": [256, 256],
    }
    return policy_kwargs


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("Testing Policy Architectures...\n")
    
    # Test learned predictor policy
    print("="*70)
    print("Testing Learned Predictor Policy")
    print("="*70)
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(120,), dtype='float32')
    
    predictor_extractor = PredictorControllerExtractor(
        obs_space, 
        features_dim=256,
        predictor_arch=[128, 128, 64],
        controller_arch=[512]
    )
    
    # Test forward pass
    test_obs = torch.randn(4, 120)
    features = predictor_extractor(test_obs)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {test_obs.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Expected: [4, 256]")
    print(f"  Status: {'✓ PASS' if features.shape == (4, 256) else '✗ FAIL'}")
    
    # Test predictor stats
    stats = predictor_extractor.get_predictor_stats()
    print(f"\nPredictor stats:")
    print(f"  Predicted target shape: {stats['predicted_target'].shape}")
    print(f"  Delayed target shape: {stats['delayed_target'].shape}")
    print(f"  Prediction offset: {stats['prediction_offset']:.4f}")
    
    # Test linear interpolation policy
    print(f"\n{'='*70}")
    print("Testing Linear Interpolation Policy")
    print("="*70)
    
    interp_extractor = LinearInterpolationExtractor(obs_space, features_dim=256)
    features_interp = interp_extractor(test_obs)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {test_obs.shape}")
    print(f"  Output shape: {features_interp.shape}")
    print(f"  Expected: [4, 256]")
    print(f"  Status: {'✓ PASS' if features_interp.shape == (4, 256) else '✗ FAIL'}")
    
    print(f"\n{'='*70}")
    print("All tests completed!")
    print(f"{'='*70}\n")