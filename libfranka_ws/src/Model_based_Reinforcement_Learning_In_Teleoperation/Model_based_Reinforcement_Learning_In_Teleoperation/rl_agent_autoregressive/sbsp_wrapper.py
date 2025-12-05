import gymnasium as gym
import torch
import numpy as np
from collections import deque
import sys
import os
import pickle
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from delay_correcting_nn import DCNN

class SBSP_Trajectory_Wrapper(gym.Wrapper):
    def __init__(self, env, n_models=5, batch_size=256, buffer_size=10000, 
                 model_path=None, stats_path=None, verbose=True, log_dir=None, 
                 save_dir="./sbsp_checkpoints", training_mode=True):  # <--- NEW FLAG
        super().__init__(env)
        self.env = env
        self.verbose = verbose
        self.save_dir = save_dir
        self.training_mode = training_mode  # <--- STORE FLAG
        
        # --- 1. SETUP LOGGING & SAVING ---
        # Only setup logging/saving if we are actually training
        if self.training_mode:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # TensorBoard
            if log_dir is None:
                log_dir = f"./sbsp_logs/run_{timestamp}"
            self.writer = SummaryWriter(log_dir=log_dir)
            
            # Checkpoints
            self.run_save_dir = os.path.join(self.save_dir, f"run_{timestamp}")
            os.makedirs(self.run_save_dir, exist_ok=True)
            
            if self.verbose:
                print(f"[SBSP] Training Mode: ON")
                print(f"[SBSP] TensorBoard: {log_dir}")
                print(f"[SBSP] Saving Models to: {self.run_save_dir}")
        else:
            self.writer = None
            if self.verbose:
                print(f"[SBSP] Training Mode: OFF (Inference Only)")

        # Dimensions
        self.robot_state_dim = 14 
        self.action_dim = 7
        
        # Hyperparameters
        self.n_models = n_models
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.start_training_threshold = 1000
        self.save_interval = 500
        
        # Normalization Stats
        self.stats = None
        self.norm_mean_in = None
        self.norm_std_in = None
        self.norm_mean_out = None
        self.norm_std_out = None
        
        # Load Stats
        if stats_path and os.path.exists(stats_path):
            if self.verbose: print(f"[SBSP] Loading normalization stats from {stats_path}")
            with open(stats_path, 'rb') as f:
                self.stats = pickle.load(f)
                try:
                    if isinstance(self.stats, dict):
                        mean = self.stats.get('mean')
                        std = self.stats.get('std')
                        if mean is not None:
                            if len(mean) >= (self.robot_state_dim + self.action_dim):
                                self.norm_mean_in = mean[:self.robot_state_dim + self.action_dim].astype(np.float32)
                                self.norm_std_in = std[:self.robot_state_dim + self.action_dim].astype(np.float32)
                                if len(mean) >= (self.robot_state_dim + self.action_dim + self.robot_state_dim):
                                    start_out = self.robot_state_dim + self.action_dim
                                    self.norm_mean_out = mean[start_out : start_out + self.robot_state_dim].astype(np.float32)
                                    self.norm_std_out = std[start_out : start_out + self.robot_state_dim].astype(np.float32)
                    if self.norm_mean_in is not None and self.verbose:
                        print("[SBSP] Normalization stats loaded successfully.")
                except Exception as e:
                    print(f"[SBSP] Warning: Could not parse stats: {e}")

        # Buffers
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.future_state_buffer = deque() 
        self.action_history = deque(maxlen=50) 
        
        self.current_prediction = None
        self.prev_robot_state = None
        self.step_counter = 0
        
        # Metrics tracking
        self.loss_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        
        # Initialize Ensemble
        self.dc_models = []
        
        # FORCE CPU for Inference
        device_load = torch.device('cpu') 
        
        pretrained_state_dict = None
        if model_path and os.path.exists(model_path):
            if self.verbose: print(f"[SBSP] Loading pre-trained weights from {model_path}")
            pretrained_state_dict = torch.load(model_path, map_location=device_load)

        for i in range(self.n_models):
            model = DCNN(
                beta=0.0003, 
                input_dims=self.robot_state_dim,
                n_actions=self.action_dim,
                layer_size=256, 
                n_layers=2
            )
            
            model.device = torch.device('cpu')
            model.to(model.device)
            
            if pretrained_state_dict:
                model.load_state_dict(pretrained_state_dict)
                
            model.eval()
            self.dc_models.append(model)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        robot_state = obs[:self.robot_state_dim].copy()
        self.prev_robot_state = robot_state
        
        self.future_state_buffer.clear()
        self.action_history.clear()
        self.step_counter = 0
        
        for _ in range(50):
            self.action_history.append(np.zeros(self.action_dim))
            
        self.current_prediction = robot_state.copy()
        obs = self._inject_prediction(obs, self.current_prediction)
        
        return obs, info

    def step(self, action):
        self.step_counter += 1
        
        self.action_history.append(action.copy())

        obs, reward, terminated, truncated, info = self.env.step(action)
        
        true_robot_state = self.env.unwrapped.get_true_current_target()[:self.robot_state_dim]
        if self.current_prediction is not None:
            sbsp_error = np.linalg.norm(self.current_prediction - true_robot_state)
            info['prediction_error'] = sbsp_error
            self.error_history.append(sbsp_error)
            
            # Log Prediction Error (Only if training_mode is ON)
            if self.training_mode and self.writer:
                self.writer.add_scalar('SBSP/Prediction_Error', sbsp_error, self.step_counter)

        delayed_robot_state = obs[:self.robot_state_dim].copy()
        
        if self.prev_robot_state is not None:
            training_pair = (
                np.append(self.prev_robot_state, action).astype(np.float32),
                delayed_robot_state.astype(np.float32)
            )
            self.replay_buffer.append(training_pair)
            
        self.prev_robot_state = delayed_robot_state

        current_delay_steps = int(info.get('current_delay_steps', 0))
        self._recalibrate(delayed_robot_state)
        
        # Rollout
        pred_state = self._ensemble_rollout(delayed_robot_state, current_delay_steps)
        
        self.future_state_buffer.append(pred_state)
        self.current_prediction = pred_state

        obs = self._inject_prediction(obs, self.current_prediction)
        
        # --- CRITICAL CHANGE: ONLY TRAIN/SAVE IF IN TRAINING MODE ---
        if self.training_mode:
            # Training Phase
            if len(self.replay_buffer) > self.start_training_threshold:
                 loss_val = self.learn()
                 if loss_val is not None:
                     self.loss_history.append(loss_val)
                     if self.writer:
                        self.writer.add_scalar('SBSP/Training_Loss', loss_val, self.step_counter)

            # Save Checkpoints
            if self.step_counter % self.save_interval == 0:
                self.save_ensemble(self.step_counter)

        # --- CONSOLE UPDATES ---
        if self.verbose and self.step_counter % 50 == 0:
            avg_loss = np.mean(self.loss_history) if self.loss_history else 0.0
            avg_err = np.mean(self.error_history) if self.error_history else 0.0
            
            # Only print buffer size if training, otherwise it's just filling up
            mode_str = "TRAIN" if self.training_mode else "EVAL"
            print(f"\r[{mode_str}] Step {self.step_counter:04d} | "
                  f"Buff: {len(self.replay_buffer)} | "
                  f"Loss: {avg_loss:.5f} | "
                  f"Err: {avg_err:.5f}", end="", flush=True)

        return obs, reward, terminated, truncated, info

    def save_ensemble(self, step):
        """Saves the first model of the ensemble as a checkpoint."""
        if not self.dc_models:
            return
            
        model_path = os.path.join(self.run_save_dir, f"sbsp_model_step_{step}.pt")
        torch.save(self.dc_models[0].state_dict(), model_path)
        
        if self.stats:
            stats_path = os.path.join(self.run_save_dir, f"sbsp_stats.pickle")
            with open(stats_path, 'wb') as f:
                pickle.dump(self.stats, f)
                
        if self.verbose:
            print(f"\n[SBSP] Saved model checkpoint to {model_path}")

    def _ensemble_rollout(self, start_state, delay_steps):
        curr_state = start_state.copy()
        
        if delay_steps <= 0:
            return curr_state
            
        delay_steps = min(delay_steps, len(self.action_history), 50)
        
        past_actions_list = list(self.action_history)
        relevant_actions = past_actions_list[-delay_steps:] 
        
        std_in_safe = self.norm_std_in + 1e-6 if self.norm_std_in is not None else 1.0
        std_out_safe = self.norm_std_out + 1e-6 if self.norm_std_out is not None else 1.0
        
        for action_t in relevant_actions:
            raw_input = np.concatenate([curr_state, action_t])
            
            if self.norm_mean_in is not None:
                model_input = (raw_input - self.norm_mean_in) / std_in_safe
                norm_obs = model_input[:self.robot_state_dim]
                norm_act = model_input[self.robot_state_dim:]
            else:
                norm_obs = curr_state
                norm_act = action_t
            
            predictions = []
            
            for model in self.dc_models:
                pred = model.predict(norm_obs, norm_act)
                predictions.append(pred)
            
            avg_pred = np.mean(predictions, axis=0)
            
            if self.norm_mean_out is not None:
                curr_state = avg_pred * std_out_safe + self.norm_mean_out
            else:
                curr_state = avg_pred
                
        return curr_state

    def _recalibrate(self, actual_arrived_state):
        if not self.future_state_buffer:
            return
        past_prediction = self.future_state_buffer.popleft()
        difference = actual_arrived_state - past_prediction
        
        for i in range(len(self.future_state_buffer)):
            self.future_state_buffer[i] += difference
        
        if self.current_prediction is not None:
             self.current_prediction += difference

    def learn(self):
        indices = np.random.choice(len(self.replay_buffer), self.batch_size)
        batch = np.array([self.replay_buffer[i] for i in indices], dtype=object)
        
        inputs = np.stack(batch[:, 0]) 
        targets = np.stack(batch[:, 1]) 
        
        if self.norm_mean_in is not None:
             inputs = (inputs - self.norm_mean_in) / (self.norm_std_in + 1e-6)
        if self.norm_mean_out is not None:
             targets = (targets - self.norm_mean_out) / (self.norm_std_out + 1e-6)

        total_loss = 0
        for model in self.dc_models:
            model.train()
            loss = model.learn(inputs, targets)
            total_loss += loss
            model.eval()
            
        return total_loss / len(self.dc_models)

    def _inject_prediction(self, obs, pred_state):
        new_obs = obs.copy()
        start_idx = -29
        end_idx = -15
        new_obs[start_idx:end_idx] = pred_state
        
        remote_state = new_obs[0:14]
        error = pred_state - remote_state
        
        err_start_idx = -15
        err_end_idx = -1
        new_obs[err_start_idx:err_end_idx] = error
        return new_obs

    def close(self):
        if self.writer:
            self.writer.close()
        return super().close()