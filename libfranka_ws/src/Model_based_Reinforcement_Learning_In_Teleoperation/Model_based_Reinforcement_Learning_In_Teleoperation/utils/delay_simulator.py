"""
Custom setting up delay patterns for teleoperation experiments.

There are three main delay configurations:
1. Low Delay: Observation delay with 40 ms variance, ~60 ms mean; Action delay fixed at 50 ms.
2. Medium Delay: Observation delay with 40 ms variance, ~100 ms mean; Action delay fixed at 50 ms.
3. High Delay: Observation delay with 40 ms variance, ~220 ms mean; Action delay fixed at 50 ms.
4. Full Range Cover: action delay fixed at 50 ms; Observation delay uniformly sampled between 40 ms and 240 ms.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import warnings
import numpy as np


class ExperimentConfig(IntEnum):
    """For mapping config to number"""
    LOW_DELAY = 1
    MEDIUM_DELAY = 2
    HIGH_DELAY = 3
    FULL_RANGE_COVER = 4
    # Debugging only:
    OBSERVATION_DELAY_ONLY = 5
    ACTION_DELAY_ONLY = 6

@dataclass(frozen=True)
class DelayParameters:
    action_delay: int
    obs_delay_min: int
    obs_delay_max: int
    name: str
    
    def __post_init__(self) -> None:
        if self.action_delay < 0:
            raise ValueError("Action delay must be non-negative")
        if self.obs_delay_min < 0 or self.obs_delay_max < 0:
            raise ValueError("Observation delays must be non-negative")
        if self.obs_delay_min > self.obs_delay_max:
            raise ValueError("obs_delay_min cannot be greater than obs_delay_max")
        
class DelaySimulator:
    
    _DELAY_CONFIGS: dict[ExperimentConfig, DelayParameters] = {
        ExperimentConfig.LOW_DELAY: DelayParameters(
            action_delay=50,
            obs_delay_min=40,
            obs_delay_max=80,
            name="Low Delay"
        ),
        ExperimentConfig.MEDIUM_DELAY: DelayParameters(
            action_delay=50,
            obs_delay_min=120,
            obs_delay_max=160,
            name="Medium Delay"
        ),
        ExperimentConfig.HIGH_DELAY: DelayParameters(
            action_delay=50,
            obs_delay_min=200,
            obs_delay_max=240,
            name="High Delay"
        ),
        ExperimentConfig.FULL_RANGE_COVER: DelayParameters(
            action_delay=50,
            obs_delay_min=40,
            obs_delay_max=240,
            name="Full Range Cover"
        ),
        ExperimentConfig.OBSERVATION_DELAY_ONLY: DelayParameters(
            action_delay=0,
            obs_delay_min=100,
            obs_delay_max=100,
            name="Observation Delay Only"
        ),
        ExperimentConfig.ACTION_DELAY_ONLY: DelayParameters(
            action_delay=50,
            obs_delay_min=0,
            obs_delay_max=0,
            name="Action Delay Only"
        ),
    }
    
    def __init__(self,
                 control_freq: int,
                 config: ExperimentConfig,
                 seed: Optional[int] = None) -> None:
        
        if control_freq <= 0:
            raise ValueError(f"control_freq must be positive, got {control_freq}")
        
        if config not in self._DELAY_CONFIGS:
            raise ValueError(f"Invalid config: {config}")
        
        self._control_freq = control_freq
        self._config = config
        self._rng = np.random.RandomState(seed)
        
        # setup delay parameters
        self._setup_delay_parameters()
        
    def _setup_delay_parameters(self) -> None:
        
        step_time_ms = 1000.0 / self._control_freq
        
        # Get configuration parameters
        params = self._DELAY_CONFIGS[self._config]
        self._config_name = params.name
        
        # Convert delays from milliseconds to discrete steps
        self._action_delay_steps = int(params.action_delay / step_time_ms)
        self._obs_delay_min_steps = int(params.obs_delay_min / step_time_ms)
        self._obs_delay_max_steps = int(params.obs_delay_max / step_time_ms)
        
        # # Ensure at least 1 step of delay (unless no-delay baseline)
        # if self._config != ExperimentConfig.NO_DELAY_BASELINE:
        #     self._obs_delay_min_steps = max(1, self._obs_delay_min_steps)
        #     self._obs_delay_max_steps = max(1, self._obs_delay_max_steps)
    
    @property
    def control_freq(self) -> int:
        return self._control_freq
    
    @property
    def config(self) -> ExperimentConfig:
        return self._config
    
    @property
    def config_name(self) -> str:
        return self._config_name
    
    def get_observation_delay(self) -> int:

        # if self._config == ExperimentConfig.NO_DELAY_BASELINE:
        #     return 0
        
        # Sample uniformly from [min, max] inclusive
        return self._rng.randint(
            self._obs_delay_min_steps,
            self._obs_delay_max_steps + 1
        )
    
    def get_action_delay(self) -> int:
        return self._action_delay_steps
    
    def get_observation_delay_steps(self, buffer_length: int) -> int:
        
        if buffer_length < 0:
            raise ValueError(f"buffer_length must be non-negative, got {buffer_length}")
        
        # # No delay baseline or empty buffer
        # if self._config == ExperimentConfig.NO_DELAY_BASELINE or buffer_length == 0:
        #     return 0
        
        # If buffer too small, return maximum possible delay
        if buffer_length <= self._obs_delay_min_steps:

            warnings.warn(
                f"Buffer length {buffer_length} insufficient for minimum "
                f"observation delay {self._obs_delay_min_steps}. "
                f"Returning reduced delay: {max(0, buffer_length - 1)}"
            )
            return max(0, buffer_length - 1)
        
        # Sample delay within buffer constraints
        max_possible_delay = min(self._obs_delay_max_steps, buffer_length - 1)
        
        return self._rng.randint(
            self._obs_delay_min_steps,
            max_possible_delay + 1
        )
    
    def get_action_delay_steps(self) -> int:
  
        # if self._config == ExperimentConfig.NO_DELAY_BASELINE:
        #     return 0
        
        return self._action_delay_steps
    
    def __repr__(self) -> str:

        return (
            f"DelaySimulator(control_freq={self._control_freq}, "
            f"config={self._config.name}, "
            f"name='{self._config_name}')"
        )