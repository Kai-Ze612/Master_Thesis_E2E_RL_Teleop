import gymnasium as gym
import torch
import numpy as np
from collections import deque
import sys
import os
from datetime import datetime  # <--- ADD THIS LINE HERE

# Import the DCNN class from your provided file
from delay_correcting_nn import DCNN

class SBSP_Trajectory_Wrapper(gym.Wrapper):
    """
    SBSP (Simulation-Based State Prediction) / PMDC Wrapper adapted for 7-DOF Teleoperation.

    Methodology:
    1. Learns a forward dynamics model (s_t, a_t -> s_{t+1}) using an Ensemble of DCNNs.
    2. Uses the learned model to roll out the delayed state to the present time.
    3. Recalibrates predictions based on the error between the oldest prediction and the newly arrived observation.
    
    Adapted from: PMDC_wrapper.py
    Target Env: TeleoperationEnvWithDelay (training_env.py)
    """

    def __init__(self, env, n_models=5, batch_size=256, buffer_size=10000):
        super().__init__(env)
        self.env = env
        
        # --- Configuration for 7-DOF Robot ---
        # The state we want to predict is q (7) + qd (7) = 14 dimensions
        self.robot_state_dim = 14 
        self.action_dim = 7
        
        # --- SBSP / PMDC Hyperparameters ---
        self.n_models = n_models
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.start_training_threshold = 1000
        
        # Experience Replay for Online Learning
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Buffer to store predictions for recalibration: (predicted_state)
        # Used to compare what we predicted N steps ago vs what just arrived
        self.future_state_buffer = deque() 
        self.current_prediction = None
        
        # Initialize Ensemble of DCNNs
        self.dc_models = []
        for i in range(self.n_models):
            # Parameters adapted for 7-DOF complexity
            model = DCNN(
                beta=0.0003, # Learning rate
                input_dims=self.robot_state_dim,
                n_actions=self.action_dim,
                layer_size=256, 
                n_layers=2
            )
            # Ensure model is in eval mode initially
            model.eval()
            self.dc_models.append(model)
            
        self.prev_robot_state = None

    def reset(self, **kwargs):
        # 1. Reset Base Environment
        obs, info = self.env.reset(**kwargs)
        
        # 2. Extract specific robot state
        robot_state = obs[:self.robot_state_dim].copy()
        self.prev_robot_state = robot_state
        
        # 3. Clear Internal Buffers
        self.future_state_buffer.clear()
        # self.replay_buffer.clear()  <--- DELETE OR COMMENT OUT THIS LINE
        
        # 4. Initialize Prediction
        self.current_prediction = robot_state.copy()
        obs = self._inject_prediction(obs, self.current_prediction)
        
        return obs, info

    def step(self, action):
        # 1. Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate Evaluation Metric
        true_robot_state = self.env.unwrapped.get_true_current_target()[:self.robot_state_dim]
        if self.current_prediction is not None:
            sbsp_error = np.linalg.norm(self.current_prediction - true_robot_state)
            info['prediction_error'] = sbsp_error

        # 2. Extract Delayed State
        delayed_robot_state = obs[:self.robot_state_dim].copy()
        
        # 3. Store Transition for Online Learning (FIXED)
        if self.prev_robot_state is not None:
            # === RESTORED LOGIC START ===
            training_pair = (
                np.append(self.prev_robot_state, action).astype(np.float32),
                delayed_robot_state.astype(np.float32)
            )
            self.replay_buffer.append(training_pair)
            # === RESTORED LOGIC END ===
            
        self.prev_robot_state = delayed_robot_state

        # 4. Recalibration and Rollout
        current_delay_steps = info.get('current_delay_steps', 0)
        self._recalibrate(delayed_robot_state)
        pred_state = self._ensemble_rollout(delayed_robot_state, action, current_delay_steps)
        
        self.future_state_buffer.append(pred_state)
        self.current_prediction = pred_state

        # 5. Injection and Learning
        obs = self._inject_prediction(obs, self.current_prediction)
        
        if len(self.replay_buffer) > self.start_training_threshold:
            self.learn()

        # Print Heartbeat
        step_count = self.env.unwrapped.current_step
        if step_count % 100 == 0:
             print(f"[{datetime.now().strftime('%H:%M:%S')}] Sim Step {step_count} / {self.env.unwrapped.max_episode_steps} | Replay Buffer: {len(self.replay_buffer)}")

        return obs, reward, terminated, truncated, info

    def _ensemble_rollout(self, start_state, action, delay_steps):
        """
        Rolls out the state 'delay_steps' into the future using the learned ensemble.
        """
        curr_state = start_state.copy()
        
        if delay_steps <= 0:
            return curr_state
            
        # Simplification: Assume Zero-Order Hold (same action) for the short delay horizon
        # This matches the logic in PMDC_wrapper.py where action is constant for rollout
        for _ in range(int(delay_steps)):
            predictions = []
            for model in self.dc_models:
                # DCNN predicts next state directly
                pred = model.predict(curr_state, action)
                predictions.append(pred)
            
            # Ensemble Mean
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