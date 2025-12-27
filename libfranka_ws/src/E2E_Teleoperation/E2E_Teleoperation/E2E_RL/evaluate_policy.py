"""
evaluate_policy.py

Script to visualize and debug a trained model checkpoint.
Run this to verify if Stage 2 (BC) or Stage 1 (Encoder) learned correctly.
"""

import torch
import numpy as np
import argparse
import time
from pathlib import Path

# Import your project modules
import E2E_Teleoperation.config.robot_config as cfg
from E2E_Teleoperation.E2E_RL.training_env import TeleoperationEnv
# NOTE: Using LSTM as per your latest file upload
from E2E_Teleoperation.E2E_RL.sac_policy_network import LSTM, JointActor 
from E2E_Teleoperation.utils.delay_simulator import ExperimentConfig
from E2E_Teleoperation.E2E_RL.local_robot_simulator import TrajectoryType

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==> Device: {device}")

    # 1. Initialize Environment (Human Render Mode)
    print(f"==> Initializing Environment (Config: HIGH_VARIANCE)...")
    env = TeleoperationEnv(
        delay_config=ExperimentConfig.HIGH_VARIANCE, # Matching your training
        trajectory_type=TrajectoryType.FIGURE_8,
        render_mode="human",  # Force GUI
        seed=42
    )

    # 2. Initialize Network
    print("==> Initializing Network Architecture...")
    # Re-create the exact architecture
    encoder = LSTM().to(device)
    actor = JointActor(encoder).to(device)

    # 3. Load Checkpoint
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Checkpoint not found at: {model_path}")
        return

    print(f"==> Loading Checkpoint: {model_path.name}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle case where checkpoint is full state dict vs just actor
        if 'actor' in checkpoint:
            actor.load_state_dict(checkpoint['actor'])
        elif 'state_dict' in checkpoint:
            actor.load_state_dict(checkpoint['state_dict'])
        else:
            # Assuming the file IS the state dict (standard for your unified_trainer)
            actor.load_state_dict(checkpoint)
            
        actor.eval() # Set to evaluation mode (no dropout, etc)
        print("==> Model Loaded Successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # 4. Run Evaluation Loop
    print("\nStarting Simulation... (Press Ctrl+C to stop)")
    print("-" * 50)
    
    num_episodes = 3
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        # Reset hidden state (if using stateful LSTM)
        hidden = None 

        while not done:
            # Prepare observation
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Forward pass (Get deterministic action)
                # Returns: mu, log_std, pred_state, next_hidden, feat
                mu, _, pred_state, next_hidden, _ = actor(obs_tensor, hidden)
                
                # Deterministic Action: Tanh(Mean) * Scale
                action = torch.tanh(mu) * actor.scale.to(device)
                action_np = action.cpu().numpy()[0]
                
                # Predictions for debugging
                pred_q = pred_state[0, :7].cpu().numpy()

            # Step Environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # Print status every 50 steps
            if step % 50 == 0:
                print(f"Ep {ep+1} Step {step} | Rew: {reward:.2f} | Err: {info['tracking_error']:.3f}")

        print(f"==> Episode {ep+1} Finished. Total Reward: {total_reward:.2f}")
        time.sleep(1.0) # Pause between episodes

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Full path to the .pth model file"
    )
    args = parser.parse_args()
    
    evaluate(args)