import torch
import numpy as np
import pickle
import os
from collections import deque
from delay_correcting_nn import DCNN

class SBSPPredictor:
    def __init__(self, model_path, stats_path, n_models=5, robot_dim=14, action_dim=7, history_len=50):
        # Force CPU to avoid device mismatch errors during single-step inference
        self.device = torch.device('cpu') 
        self.robot_dim = robot_dim
        self.action_dim = action_dim
        self.history_len = history_len
        
        # --- ROBUST STATS LOADING ---
        print(f"[SBSP] Loading stats from {stats_path}")
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            
        # Try to find Mean
        if 'mean' in stats: self.mean_in = stats['mean']
        elif 'running_mean' in stats: self.mean_in = stats['running_mean']
        elif 'mu' in stats: self.mean_in = stats['mu']
        else:
            print("[WARNING] Could not find 'mean' in stats. Using 0.0")
            self.mean_in = np.zeros(robot_dim + action_dim, dtype=np.float32)

        # Try to find Std
        if 'std' in stats: self.std_in = stats['std']
        elif 'running_std' in stats: self.std_in = stats['running_std']
        elif 'scale' in stats: self.std_in = stats['scale']
        elif 'sigma' in stats: self.std_in = stats['sigma']
        else:
            print("[WARNING] Could not find 'std' in stats. Using 1.0")
            self.std_in = np.ones(robot_dim + action_dim, dtype=np.float32)

        # Ensure types and shapes
        in_dim = robot_dim + action_dim
        self.mean_in = np.array(self.mean_in).flatten()
        self.std_in = np.array(self.std_in).flatten()
        
        if len(self.mean_in) >= in_dim:
            self.mean_in = self.mean_in[:in_dim].astype(np.float32)
            self.std_in = self.std_in[:in_dim].astype(np.float32)
        else:
            pad = in_dim - len(self.mean_in)
            self.mean_in = np.pad(self.mean_in, (0, pad))
            self.std_in = np.pad(self.std_in, (0, pad), constant_values=1.0)

        self.std_in += 1e-6
        self.mean_out = self.mean_in[:robot_dim]
        self.std_out = self.std_in[:robot_dim]

        # Load Models to CPU
        print(f"[SBSP] Loading models from {model_path}")
        self.models = []
        # Ensure map_location is set to CPU
        state_dict = torch.load(model_path, map_location=self.device)
        
        for _ in range(n_models):
            model = DCNN(beta=0.0, input_dims=robot_dim, n_actions=action_dim, layer_size=256, n_layers=2)
            model.to(self.device) # Move model to CPU
            
            # --- ADD THIS LINE ---
            model.device = self.device # Force the DCNN class to know it is on CPU
            # ---------------------

            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)
            
        self.action_history = deque(maxlen=history_len)
        for _ in range(history_len):
            self.action_history.append(np.zeros(action_dim, dtype=np.float32))
            
    def push_action(self, action):
        self.action_history.append(action.copy())
        
    def predict(self, current_delayed_state, delay_steps):
        curr_state = current_delayed_state.copy()
        
        if delay_steps <= 0: return curr_state
            
        steps_to_roll = min(delay_steps, len(self.action_history))
        relevant_actions = list(self.action_history)[-steps_to_roll:]
        
        for action in relevant_actions:
            raw_input = np.concatenate([curr_state, action])
            if not np.isfinite(raw_input).all():
                raw_input = np.nan_to_num(raw_input)
                
            norm_input = (raw_input - self.mean_in) / self.std_in
            norm_obs = norm_input[:self.robot_dim]
            norm_act = norm_input[self.robot_dim:]
            
            preds = []
            for model in self.models:
                with torch.no_grad():
                    # DCNN.predict might try to move things to its own device
                    # We bypass that by calling the model directly if needed, or ensuring predict handles CPU
                    # Assuming DCNN.predict handles numpy -> tensor conversion:
                    p = model.predict(norm_obs, norm_act)
                preds.append(p)
            
            avg_pred_norm = np.mean(preds, axis=0)
            curr_state = avg_pred_norm * self.std_out + self.mean_out
            
        return curr_state