import gymnasium as gym
import torch
import numpy as np
from collections import deque
import sys
import os
from datetime import datetime

from delay_correcting_nn import DCNN

class SBSP_Trajectory_Wrapper(gym.Wrapper):
    def __init__(self, env, n_models=5, batch_size=256, buffer_size=10000):
        super().__init__(env)
        self.env = env
        
        # Dimensions
        self.robot_state_dim = 14 
        self.action_dim = 7
        
        # Hyperparameters
        self.n_models = n_models
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.start_training_threshold = 1000
        
        # Buffers
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.future_state_buffer = deque() 
        
        # --- FIX 1: Action History Buffer ---
        # We need to store enough actions to cover the maximum possible delay
        # Assuming max delay is around 1 sec @ 20Hz = 20 steps. 50 is safe.
        self.action_history = deque(maxlen=50) 
        
        self.current_prediction = None
        self.prev_robot_state = None
        
        # Initialize Ensemble
        self.dc_models = []
        for i in range(self.n_models):
            model = DCNN(
                beta=0.0003, 
                input_dims=self.robot_state_dim,
                n_actions=self.action_dim,
                layer_size=256, 
                n_layers=2
            )
            model.eval()
            self.dc_models.append(model)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        robot_state = obs[:self.robot_state_dim].copy()
        self.prev_robot_state = robot_state
        
        self.future_state_buffer.clear()
        self.action_history.clear() # Clear action history
        
        # Prefill action history with zeros (or neutral position)
        for _ in range(50):
            self.action_history.append(np.zeros(self.action_dim))
            
        self.current_prediction = robot_state.copy()
        obs = self._inject_prediction(obs, self.current_prediction)
        
        return obs, info

    def step(self, action):
        # --- FIX 1: Store action immediately ---
        self.action_history.append(action.copy())

        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Evaluation Metric
        true_robot_state = self.env.unwrapped.get_true_current_target()[:self.robot_state_dim]
        if self.current_prediction is not None:
            sbsp_error = np.linalg.norm(self.current_prediction - true_robot_state)
            info['prediction_error'] = sbsp_error

        delayed_robot_state = obs[:self.robot_state_dim].copy()
        
        # Online Learning Storage
        if self.prev_robot_state is not None:
            # We need the action that caused prev_state -> delayed_state.
            # This logic depends on the specific delay implementation, 
            # but for 1-step prediction training, using the previous action is standard.
            # However, simpler is to just use the action executed *at that time*.
            # For simplicity in this wrapper, we assume the environment is Markovian 
            # enough that (s_t, a_t -> s_{t+1}).
            
            # Note: You might need to retrieve the specific action associated with this transition
            # if your environment provides it, but usually 'action' (current) is NOT the one
            # associated with the delayed state transition. 
            
            # CRITICAL: For training the DCNN (One-step model), we strictly need:
            # State(t-1), Action(t-1) -> State(t)
            # But here we are receiving State(t-k). 
            # Since we don't easily know Action(t-k-1) without complex indexing, 
            # many PMDC implementations train on the *current* transition if available,
            # or they store (s, a) pairs in a buffer and pop them when the delay resolves.
            
            # YOUR IMPLEMENTATION (Original): 
            # Uses 'action' (current) with 'prev_robot_state' (delayed). 
            # This trains the model to think: "Delayed State + Current Action = Next Delayed State".
            # This is roughly correct assuming the delay is consistent between steps.
            
            training_pair = (
                np.append(self.prev_robot_state, action).astype(np.float32),
                delayed_robot_state.astype(np.float32)
            )
            self.replay_buffer.append(training_pair)
            
        self.prev_robot_state = delayed_robot_state

        # Recalibration and Rollout
        current_delay_steps = int(info.get('current_delay_steps', 0))
        self._recalibrate(delayed_robot_state)
        
        # --- FIX 2: Pass delay steps to rollout ---
        pred_state = self._ensemble_rollout(delayed_robot_state, current_delay_steps)
        
        self.future_state_buffer.append(pred_state)
        self.current_prediction = pred_state

        obs = self._inject_prediction(obs, self.current_prediction)
        
        if len(self.replay_buffer) > self.start_training_threshold:
            self.learn()

        return obs, reward, terminated, truncated, info

    def _ensemble_rollout(self, start_state, delay_steps):
        """
        Rolls out the state using the history of actions.
        """
        curr_state = start_state.copy()
        
        if delay_steps <= 0:
            return curr_state
            
        # --- FIX 3: Replay specific past actions ---
        # If delay is k, we need the last k actions from history
        # actions_to_apply = [a_{t-k}, a_{t-k+1}, ..., a_{t-1}]
        
        # Guard against delay being larger than history
        delay_steps = min(delay_steps, len(self.action_history))
        
        # Slice the deque to get the relevant past actions
        # converting to list is O(N), but N is small (e.g. 10-20)
        past_actions_list = list(self.action_history)
        relevant_actions = past_actions_list[-delay_steps:] 
        
        for action_t in relevant_actions:
            predictions = []
            for model in self.dc_models:
                # Use the HISTORICAL action, not the current one
                pred = model.predict(curr_state, action_t)
                predictions.append(pred)
            
            curr_state = np.mean(predictions, axis=0)
            
        return curr_state

    def _recalibrate(self, actual_arrived_state):
        """
        SBSP Recalibration: 
        Calculate error between what we predicted steps ago (popped from buffer) 
        and what just arrived (actual_arrived_state).
        """
        if not self.future_state_buffer:
            return

        # Pop the oldest prediction
        past_prediction = self.future_state_buffer.popleft()
        
        # Calculate Error
        difference = actual_arrived_state - past_prediction
        
        # Apply Correction to all future predictions currently in buffer
        # This shifts the entire trajectory to align with the latest ground truth
        updated_buffer = []
        for pred in self.future_state_buffer:
            updated_buffer.append(pred + difference)
            
        self.future_state_buffer = deque(updated_buffer)
        
        # Apply correction to current cached prediction
        if self.current_prediction is not None:
             self.current_prediction += difference

    def learn(self):
        """Trains the ensemble on the replay buffer."""
        indices = np.random.choice(len(self.replay_buffer), self.batch_size)
        batch = np.array([self.replay_buffer[i] for i in indices], dtype=object)
        
        inputs = np.stack(batch[:, 0]) # State + Action
        targets = np.stack(batch[:, 1]) # Next State
        
        for model in self.dc_models:
            model.train()
            model.learn(inputs, targets)
            model.eval()

    def _inject_prediction(self, obs, pred_state):
        """
        Injects the SBSP prediction into the 112D observation vector.
        
        Structure from training_env.py:
        [remote(14), history(...), PRED(14), ERROR(14), DELAY(1)]
        indices: -29:-15 for PRED, -15:-1 for ERROR.
        """
        new_obs = obs.copy()
        
        # 1. Update Prediction (pred_q + pred_qd)
        start_idx = -29
        end_idx = -15
        new_obs[start_idx:end_idx] = pred_state
        
        # 2. Update Error (Pred - Remote)
        # Remote state is at the beginning of obs
        remote_state = new_obs[0:14]
        error = pred_state - remote_state
        
        err_start_idx = -15
        err_end_idx = -1
        new_obs[err_start_idx:err_end_idx] = error
        
        return new_obs