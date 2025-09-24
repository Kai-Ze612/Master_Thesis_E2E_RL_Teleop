"""
A delay simulator to simulate different delay patterns
"""
import numpy as np

class DelaySimulator:
    def __init__(self, control_freq: int, experiment_config: int):
        self.control_freq = control_freq
        self.experiment_config = experiment_config
        self._setup_delay_parameters()
    
    def _setup_delay_parameters(self):
        """Configures delay parameters based on the chosen experiment configuration."""
        step_time_ms = 1000 / self.control_freq
        
        delay_configs = {
            1: {"action_ms": 50, "obs_min_ms": 40, "obs_max_ms": 80, "name": "Low Delay"},  # 40 ms variance, ~60 ms mean
            2: {"action_ms": 50, "obs_min_ms": 120, "obs_max_ms": 160, "name": "Medium Delay"}, # 40 ms variance, ~100 ms mean
            3: {"action_ms": 50, "obs_min_ms": 200, "obs_max_ms": 240, "name": "High Delay"},  # 40 ms variance, ~220 ms mean
            4: {"action_ms": 0, "obs_min_ms": 0, "obs_max_ms": 0, "name": "No Delay Baseline"},              # Reference
            
            ## For debugging only:
            5: {"action_ms": 0,   "obs_min_ms": 100, "obs_max_ms": 100, "name": "Observation Delay ONLY"}, # No action delay, fixed 100 ms obs delay
            6: {"action_ms": 50,  "obs_min_ms": 0,   "obs_max_ms": 0,   "name": "Action Delay ONLY"} # No obs delay, fixed 50 ms action delay
        }
       
        if self.experiment_config not in delay_configs:
            raise ValueError(f"Invalid experiment_config: {self.experiment_config}")
        config = delay_configs[self.experiment_config]
        
        # Store config name for reporting
        self.delay_config_name = config["name"]
        
        # Convert all delay parameters from milliseconds to discrete simulation steps
        self.constant_action_delay = int(config["action_ms"] / step_time_ms)
        self.stochastic_obs_delay_min = int(config["obs_min_ms"] / step_time_ms)
        self.stochastic_obs_delay_max = int(config["obs_max_ms"] / step_time_ms)
        
        # Ensure there's at least 1 step of delay unless it's the no-delay case
        if self.experiment_config != 4:  # Config 4 is no delay baseline
            self.stochastic_obs_delay_min = max(1, self.stochastic_obs_delay_min)
            self.stochastic_obs_delay_max = max(1, self.stochastic_obs_delay_max)
    
    def get_observation_delay(self) -> int:
        """Samples and returns a stochastic observation delay in steps."""
        if self.experiment_config == 4:  # No delay baseline
            return 0
        return np.random.randint(self.stochastic_obs_delay_min, self.stochastic_obs_delay_max + 1)
        
    def get_action_delay(self) -> int:
        """Returns the constant action delay in steps."""
        return self.constant_action_delay
    
    def get_observation_delay_steps(self, buffer_length: int) -> int:
        """Get observation delay steps matching RL environment logic exactly."""
        if self.experiment_config == 4 or buffer_length == 0:  # No delay baseline
            return 0  # No delay for baseline or empty buffer
            
        if buffer_length <= self.stochastic_obs_delay_min:
            return max(0, buffer_length - 1)  # Ensure non-negative
            
        # Sample delay exactly like RL environment
        max_possible_delay = min(self.stochastic_obs_delay_max, buffer_length - 1)
        delay_steps = np.random.randint(self.stochastic_obs_delay_min, max_possible_delay + 1)
        return delay_steps
        
    def get_action_delay_steps(self, buffer_length: int) -> int:
        """Get action delay steps matching RL environment logic exactly."""
        if self.experiment_config == 4:  # No delay baseline
            return 0  # No delay for baseline
            
        # Return constant action delay
        return self.constant_action_delay