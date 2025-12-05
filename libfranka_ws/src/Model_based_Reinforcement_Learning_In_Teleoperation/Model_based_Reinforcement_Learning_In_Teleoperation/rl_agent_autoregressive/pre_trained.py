import os
import pickle
import numpy as np
import torch
import gymnasium as gym

# Import your environment and model
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from delay_correcting_nn import DCNN

def generate_dataset(dataset_path, n_steps=50000):
    print(f"--- Generating {n_steps} steps of data for pretraining ---")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # 1. Initialize with explicit render_mode=None to prevent viewer hangs
    print("[1/4] Initializing Environment...")
    env = TeleoperationEnvWithDelay(
        delay_config=ExperimentConfig.LOW_DELAY,
        randomize_trajectory=True,
        render_mode=None  # <--- FORCE NO RENDER
    )
    
    data = []
    
    # 2. Add print before reset
    print("[2/4] Resetting Environment... (This runs pre-computation and may take 10-20 seconds)")
    obs, _ = env.reset()
    print("[3/4] Reset Complete. Starting Data Collection...")
    
    for i in range(n_steps):
        action = env.action_space.sample()
        
        # Step the environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        robot_state = obs[:14]
        transition = np.concatenate([robot_state, action])
        data.append(transition)
        
        obs = next_obs
        
        if terminated or truncated:
            # Print a small dot or message when an episode finishes
            print(f"    Episode finished at step {i+1}. Resetting...")
            obs, _ = env.reset()
            
        # 3. Print more frequently (every 100 steps instead of 5000)
        if (i+1) % 100 == 0:
            print(f"[4/4] Collected {i+1}/{n_steps} steps...")
            
    with open(dataset_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Dataset saved to {dataset_path}")
    env.close()
    return np.array(data)

def train_model(dataset_path, model_save_path, epochs=5, batch_size=256):
    """
    Trains the DCNN using the generated dataset.
    """
    print("--- Starting Training ---")
    
    # Load Data
    with open(dataset_path, "rb") as f:
        raw_data = pickle.load(f)
    dataset = np.array(raw_data)
    
    # Configuration for 7-DOF Robot
    obs_space = 14  # q(7) + qd(7)
    act_space = 7   # torque(7)
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCNN(
        beta=0.0003, 
        input_dims=obs_space, 
        n_actions=act_space, 
        layer_size=256, 
        n_layers=2
    ).to(device)
    
    # Prepare Data for 1-step prediction
    # Input: State_t + Action_t
    # Label: State_{t+1}
    
    X = dataset[:-1] # Inputs (All steps except the last one)
    Y = dataset[1:, :obs_space] # Targets (All steps except first, only state)
    
    # Simple Training Loop
    model.train()
    n_samples = len(X)
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(n_samples)
        
        for i in range(n_batches):
            idx = indices[i*batch_size : (i+1)*batch_size]
            batch_x = X[idx]
            batch_y = Y[idx]
            
            loss = model.learn(batch_x, batch_y)
            epoch_loss += loss
            
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")
        
    # Save Model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Pretrained model saved to: {model_save_path}")

if __name__ == "__main__":
    # Define paths
    DATASET_FILE = "./dataset/Teleoperation_7DOF/traj_data.pickle"
    MODEL_FILE = "./models/FetchPush-RemotePDNorm-v0/2-256-1_step_prediction_sd.pt" # Matches your wrapper path
    
    # 1. Generate Data if it doesn't exist
    if not os.path.exists(DATASET_FILE):
        generate_dataset(DATASET_FILE)
    else:
        print("Dataset found. Skipping generation.")
        
    # 2. Train Model
    train_model(DATASET_FILE, MODEL_FILE)