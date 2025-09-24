import numpy as np
import os
import json
from tqdm import tqdm

from Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays.utils.delay_simulator import DelaySimulator
from Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays.rl_agent.local_robot_simulator import LocalRobotSimulator

class LSTMDataGenerator:
    """Generate LSTM training data using your LocalRobotSimulator and DelaySimulator"""
    
    def __init__(self, control_freq: int = 200):
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        
        # Call the local robot simulator for leader state generation
        self.leader = LocalRobotSimulator(control_freq=control_freq)
        
    def generate_trajectory_with_delays(self, experiment_config: int, trajectory_duration: int = 10000):
        
        # Initialize delay simulator
        delay_simulator = DelaySimulator(self.control_freq, experiment_config)
        
        # Reset leader robot with random trajectory parameters
        leader_start_pos, info = self.leader.reset()
        
        # Storage for trajectory data
        leader_positions = [leader_start_pos]
        delays = []
        delayed_positions = []
        
        # Simulate trajectory
        for step in range(trajectory_duration):
            # Get next leader position
            new_leader_pos, _, _, _, _ = self.leader.step()
            leader_positions.append(new_leader_pos)
            
            # Calculate delay for this step
            buffer_length = len(leader_positions)
            obs_delay_steps = delay_simulator.get_observation_delay_steps(buffer_length)
            
            # Get delayed position (what the follower would see)
            delay_index = len(leader_positions) - 1 - obs_delay_steps
            delayed_pos = leader_positions[max(0, delay_index)]
            
            delays.append(obs_delay_steps)
            delayed_positions.append(delayed_pos)
        
        return {
            'leader_positions': np.array(leader_positions),
            'delayed_positions': np.array(delayed_positions),
            'delays': np.array(delays),
            'experiment_config': experiment_config,
            'trajectory_params': info,
            'trajectory_length': len(leader_positions)
        }
    
    def create_lstm_training_samples(self, trajectory_data, sequence_length: int = 20):
        """
        Convert trajectory data into LSTM training samples
        
        Args:
            trajectory_data: Output from generate_trajectory_with_delays()
            sequence_length: Length of input sequences for LSTM
            
        Returns:
            Dictionary with input sequences and targets
        """
        
        leader_positions = trajectory_data['leader_positions']
        delayed_positions = trajectory_data['delayed_positions']
        delays = trajectory_data['delays']
        
        input_sequences = []
        target_positions = []
        delay_values = []
        
        # Create overlapping sequences
        for i in range(sequence_length, len(delayed_positions)):
            # Input: sequence of delayed positions
            input_seq = delayed_positions[i-sequence_length:i]
            
            # Target: current real leader position
            target_pos = leader_positions[i]
            
            # Delay for this sample
            delay_val = delays[i]
            
            # Only include samples with meaningful delays
            if delay_val > 0:
                input_sequences.append(input_seq)
                target_positions.append(target_pos)
                delay_values.append(delay_val)
        
        return {
            'input_sequences': np.array(input_sequences),
            'target_positions': np.array(target_positions),
            'delays': np.array(delay_values),
            'sequence_length': sequence_length
        }

# Function to generate complete dataset
def generate_lstm_dataset(num_trajectories_per_config: int = 200,
                         experiment_configs: list = [1, 2, 3, 4],
                         trajectory_duration: int = 10000,
                         sequence_length: int = 20,
                         save_path: str = './lstm_data/'):

    print(f"Generating LSTM training dataset...")
    print(f"Trajectories per config: {num_trajectories_per_config}")
    print(f"Experiment configs: {experiment_configs}")
    print(f"Trajectory duration: {trajectory_duration} steps")
    print(f"Sequence length: {sequence_length}")
    
    # Initialize data generator
    generator = LSTMDataGenerator()
    
    # Storage for all data
    all_input_sequences = []
    all_target_positions = []
    all_delays = []
    all_experiment_configs = []
    all_trajectory_params = []
    
    delay_stats = {}
    total_trajectories = len(experiment_configs) * num_trajectories_per_config
    
    with tqdm(total=total_trajectories, desc="Generating trajectories") as pbar:
        
        for config_id in experiment_configs:
            print(f"\nProcessing delay config {config_id}")
            
            config_samples = 0
            config_delays = []
            
            for traj_idx in range(num_trajectories_per_config):
                try:
                    # Generate trajectory with delays
                    traj_data = generator.generate_trajectory_with_delays(
                        experiment_config=config_id,
                        trajectory_duration=trajectory_duration
                    )
                    
                    # Convert to LSTM training samples
                    lstm_samples = generator.create_lstm_training_samples(
                        traj_data, sequence_length=sequence_length
                    )
                    
                    # Only keep trajectories with enough samples
                    if len(lstm_samples['input_sequences']) > 10:
                        all_input_sequences.extend(lstm_samples['input_sequences'])
                        all_target_positions.extend(lstm_samples['target_positions'])
                        all_delays.extend(lstm_samples['delays'])
                        all_experiment_configs.extend([config_id] * len(lstm_samples['delays']))
                        all_trajectory_params.extend([traj_data['trajectory_params']] * len(lstm_samples['delays']))
                        
                        config_samples += len(lstm_samples['input_sequences'])
                        config_delays.extend(lstm_samples['delays'])
                    
                except Exception as e:
                    print(f"Trajectory {traj_idx} failed: {e}")
                    continue
                
                pbar.update(1)
            
            # Store statistics
            if config_delays:
                delay_stats[config_id] = {
                    'trajectories': num_trajectories_per_config,
                    'total_samples': config_samples,
                    'delay_min': int(np.min(config_delays)),
                    'delay_max': int(np.max(config_delays)),
                    'delay_mean': float(np.mean(config_delays)),
                    'delay_std': float(np.std(config_delays))
                }
                
                print(f"Config {config_id}: {config_samples} samples, "
                      f"delay: {np.mean(config_delays):.1f}±{np.std(config_delays):.1f} steps")
    
    # Convert to numpy arrays
    input_sequences = np.array(all_input_sequences)
    target_positions = np.array(all_target_positions)
    delays = np.array(all_delays)
    experiment_configs = np.array(all_experiment_configs)

    # Create train/validation split
    total_samples = len(input_sequences)
    train_split = int(0.8 * total_samples)
    
    # Shuffle data
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]
    
    # Split data
    train_data = {
        'input_sequences': input_sequences[train_indices],
        'target_positions': target_positions[train_indices],
        'delays': delays[train_indices],
        'experiment_configs': experiment_configs[train_indices]
    }
    
    val_data = {
        'input_sequences': input_sequences[val_indices],
        'target_positions': target_positions[val_indices],
        'delays': delays[val_indices],
        'experiment_configs': experiment_configs[val_indices]
    }
    
    # Save data
    os.makedirs(save_path, exist_ok=True)
    
    # Save training and validation sets
    np.save(os.path.join(save_path, 'train_data.npy'), train_data)
    np.save(os.path.join(save_path, 'val_data.npy'), val_data)
    
    # Save complete dataset
    complete_data = {
        'input_sequences': input_sequences,
        'target_positions': target_positions,
        'delays': delays,
        'experiment_configs': experiment_configs,
        'trajectory_params': all_trajectory_params
    }
    np.save(os.path.join(save_path, 'complete_data.npy'), complete_data)
    
    # Save metadata
    metadata = {
        'total_samples': int(total_samples),
        'train_samples': int(len(train_indices)),
        'val_samples': int(len(val_indices)),
        'input_shape': list(input_sequences.shape),
        'target_shape': list(target_positions.shape),
        'sequence_length': sequence_length,
        'experiment_configs': experiment_configs.tolist(),
        'delay_statistics': delay_stats,
        'generation_params': {
            'num_trajectories_per_config': num_trajectories_per_config,
            'trajectory_duration': trajectory_duration,
            'control_frequency': 200
        }
    }
    
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nData saved to {save_path}")
    print(f"Files created:")
    print(f"  - train_data.npy ({len(train_indices):,} samples)")
    print(f"  - val_data.npy ({len(val_indices):,} samples)")
    print(f"  - complete_data.npy ({total_samples:,} samples)")
    print(f"  - metadata.json")
    
    return complete_data, delay_stats

# Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("LSTM TRAINING DATA GENERATION")
    print("=" * 60)
    
    # Generate LSTM training data
    data, stats = generate_lstm_dataset(
        num_trajectories_per_config=200,
        experiment_configs=[3],
        trajectory_duration=10000,
        sequence_length=20,
        save_path='./lstm_data/'
    )

    print("\nLSTM data generation completed!")