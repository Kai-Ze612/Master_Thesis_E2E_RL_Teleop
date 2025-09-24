# import numpy as np
# import os

# class DelaySimulator:
#     """
#     A delay simulator to simulate different delay patterns based on predefined configurations.
#     """
#     def __init__(self, control_freq: int, experiment_config: int):
#         self.control_freq = control_freq
#         self.experiment_config = experiment_config
#         self._setup_delay_parameters()

#     def _setup_delay_parameters(self):
#         """Configures delay parameters based on the chosen experiment configuration."""
#         step_time_ms = 1000 / self.control_freq
        
#         delay_configs = {
#             1: {"action_ms": 50, "obs_min_ms": 40, "obs_max_ms": 80, "name": "Low Delay"},
#             2: {"action_ms": 50, "obs_min_ms": 120, "obs_max_ms": 160, "name": "Medium Delay"},
#             3: {"action_ms": 50, "obs_min_ms": 200, "obs_max_ms": 240, "name": "High Delay"},
#             4: {"action_ms": 0, "obs_min_ms": 0, "obs_max_ms": 0, "name": "No Delay Baseline"},
#             5: {"action_ms": 0, "obs_min_ms": 100, "obs_max_ms": 100, "name": "Observation Delay ONLY"},
#             6: {"action_ms": 50, "obs_min_ms": 0, "obs_max_ms": 0, "name": "Action Delay ONLY"}
#         }
        
#         if self.experiment_config not in delay_configs:
#             raise ValueError(f"Invalid experiment_config: {self.experiment_config}")

#         config = delay_configs[self.experiment_config]
        
#         self.constant_action_delay = int(round(config["action_ms"] / step_time_ms))
#         self.stochastic_obs_delay_min = int(round(config["obs_min_ms"] / step_time_ms))
#         self.stochastic_obs_delay_max = int(round(config["obs_max_ms"] / step_time_ms))

#     def get_observation_delay(self) -> int:
#         """Samples and returns a stochastic observation delay in steps."""
#         if self.stochastic_obs_delay_min >= self.stochastic_obs_delay_max:
#              return self.stochastic_obs_delay_min
#         return np.random.randint(self.stochastic_obs_delay_min, self.stochastic_obs_delay_max + 1)
        
#     def get_action_delay(self) -> int:
#         """Returns the constant action delay in steps."""
#         return self.constant_action_delay

# def generate_data(num_samples_per_class: int, sequence_length: int, control_freq: int):
#     """
#     Generates a dataset of delay sequences for LSTM training.
    
#     Args:
#         num_samples_per_class: The number of sequences to generate for each delay class.
#         sequence_length: The length of each individual sequence (time steps).
#         control_freq: The control frequency in Hz.

#     Returns:
#         A tuple (X, y) containing the feature matrix and label vector.
#     """
#     all_features = []
#     all_labels = []
    
#     # Configurations to classify: 1:Low, 2:Medium, 3:High
#     target_configs = {1: 0, 2: 1, 3: 2} # Mapping from config ID to class label

#     for config_id, label in target_configs.items():
#         print(f"Generating data for config {config_id} (Label {label})...")
#         simulator = DelaySimulator(control_freq=control_freq, experiment_config=config_id)
        
#         # Generate samples for the current class
#         for _ in range(num_samples_per_class):
#             sequence = []
#             for _ in range(sequence_length):
#                 action_delay = simulator.get_action_delay()
#                 obs_delay = simulator.get_observation_delay()
#                 total_delay = action_delay + obs_delay
#                 sequence.append(total_delay)
#             all_features.append(sequence)
#             all_labels.append(label)

#     # Convert to NumPy arrays
#     X = np.array(all_features, dtype=np.float32)
#     y = np.array(all_labels, dtype=np.int64)
    
#     # Add a feature dimension for LSTM input: (samples, seq_len, features)
#     X = np.expand_dims(X, axis=2)

#     # Shuffle the dataset
#     indices = np.arange(X.shape[0])
#     np.random.shuffle(indices)
#     X = X[indices]
#     y = y[indices]
    
#     return X, y

# def main():
#     """
#     Main function to generate and save the delay data.
#     """
#     # --- Data Generation Parameters ---
#     NUM_SAMPLES_PER_CLASS = 2000
#     SEQUENCE_LENGTH = 50
#     CONTROL_FREQ = 100 # 100 Hz, meaning 1 step = 10 ms
#     OUTPUT_FILENAME = 'delay_training_data.npz'

#     # --- Generate the Dataset ---
#     X_data, y_data = generate_data(
#         num_samples_per_class=NUM_SAMPLES_PER_CLASS,
#         sequence_length=SEQUENCE_LENGTH,
#         control_freq=CONTROL_FREQ
#     )

#     print(f"\nGenerated data shape: X={X_data.shape}, y={y_data.shape}")

#     # --- Save the Data to a File ---
#     np.savez_compressed(OUTPUT_FILENAME, X=X_data, y=y_data)
    
#     print(f"✅ Data successfully saved to {os.path.abspath(OUTPUT_FILENAME)}")


# if __name__ == '__main__':
#     main()
    
    
import numpy as np
import os

# The DelaySimulator class is the same as before.
class DelaySimulator:
    def __init__(self, control_freq: int, experiment_config: int):
        self.control_freq = control_freq
        self.experiment_config = experiment_config
        self._setup_delay_parameters()

    def _setup_delay_parameters(self):
        step_time_ms = 1000 / self.control_freq
        delay_configs = {
            1: {"action_ms": 50, "obs_min_ms": 40, "obs_max_ms": 80, "name": "Low Delay"},
            2: {"action_ms": 50, "obs_min_ms": 120, "obs_max_ms": 160, "name": "Medium Delay"},
            3: {"action_ms": 50, "obs_min_ms": 200, "obs_max_ms": 240, "name": "High Delay"},
        }
        config = delay_configs[self.experiment_config]
        self.constant_action_delay = int(round(config["action_ms"] / step_time_ms))
        self.stochastic_obs_delay_min = int(round(config["obs_min_ms"] / step_time_ms))
        self.stochastic_obs_delay_max = int(round(config["obs_max_ms"] / step_time_ms))

    def get_observation_delay(self) -> int:
        if self.stochastic_obs_delay_min >= self.stochastic_obs_delay_max:
             return self.stochastic_obs_delay_min
        return np.random.randint(self.stochastic_obs_delay_min, self.stochastic_obs_delay_max + 1)
        
    def get_action_delay(self) -> int:
        return self.constant_action_delay

def main():
    # --- Parameters ---
    NUM_SAMPLES_PER_CLASS = 2000
    SEQUENCE_LENGTH = 50
    CONTROL_FREQ = 100
    OUTPUT_FILENAME = 'delay_features_data.npz' # Use a new filename

    all_features = []
    all_labels = []
    
    target_configs = {1: 0, 2: 1, 3: 2} # Low: 0, Medium: 1, High: 2

    for config_id, label in target_configs.items():
        print(f"Generating features for config {config_id}...")
        simulator = DelaySimulator(control_freq=CONTROL_FREQ, experiment_config=config_id)
        for _ in range(NUM_SAMPLES_PER_CLASS):
            sequence = [simulator.get_action_delay() + simulator.get_observation_delay() for _ in range(SEQUENCE_LENGTH)]
            
            # --- FEATURE ENGINEERING ---
            # Instead of saving the sequence, calculate statistics
            mean_val = np.mean(sequence)
            std_val = np.std(sequence)
            min_val = np.min(sequence)
            max_val = np.max(sequence)
            all_features.append([mean_val, std_val, min_val, max_val])
            all_labels.append(label)

    # Convert to NumPy arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    # Shuffle the dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Save the engineered features
    np.savez_compressed(OUTPUT_FILENAME, X=X, y=y)
    
    print(f"\nGenerated feature data shape: X={X.shape}, y={y.shape}")
    print(f"✅ Feature data saved to {os.path.abspath(OUTPUT_FILENAME)}")

if __name__ == '__main__':
    main()