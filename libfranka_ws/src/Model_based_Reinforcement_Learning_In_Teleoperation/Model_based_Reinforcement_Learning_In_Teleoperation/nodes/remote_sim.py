"""
SBSP-RL Teleoperation Deployment Script (Modified from PD Baseline)
- The Remote Robot tracks the SBSP-predicted Leader State (delay compensated).
- The Remote Robot's PD controller is augmented with torque compensation (tau_RL) from the trained SAC agent.
- All components (Leader, Remote, Delay, SBSP) are integrated into the Gymnasium environment wrapper for a clean step() interface.
"""
import numpy as np
import time
import mujoco
import mujoco.viewer
from collections import deque
from dataclasses import dataclass, field
import torch
import os
import sys
import pickle
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- Project Imports ---
# NOTE: Ensure these imports correctly point to your local files
try:
    # Use relative imports that should work within the project's root structure
    from utils.delay_simulator import DelaySimulator, ExperimentConfig
    from config.robot_config import DEFAULT_CONTROL_FREQ, N_JOINTS, CHECKPOINT_DIR_RL, WARM_UP_DURATION
    from rl_agent_autoregressive.training_env import TeleoperationEnvWithDelay
    from rl_agent_autoregressive.sbsp_wrapper import SBSP_Trajectory_Wrapper
    from rl_agent_autoregressive.delay_correcting_nn import DCNN
except ImportError as e:
    print(f"Error importing project files: {e}")
    print("Ensure you run this script from the project root directory or that the PYTHONPATH is set correctly.")
    sys.exit(1)
# --- Stable-Baselines3 SAC Import ---
from stable_baselines3 import SAC
# --- ROS Imports (as in baseline) ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
# =============================================================================
# PUBLISH RATE CONFIGURATION
# =============================================================================
EE_POSE_PUBLISH_FREQ = 50
METRICS_PUBLISH_FREQ = 50
# =============================================================================
# DCNN ENSEMBLE LOADING FUNCTION
# =============================================================================
def load_dcnn_ensemble(wrapper: SBSP_Trajectory_Wrapper, model_dir: str, n_models: int = 5):
    """
    Loads trained DCNN weights into the SBSP wrapper's ensemble.
  
    Args:
        wrapper: The initialized SBSP_Trajectory_Wrapper instance.
        model_dir: Directory containing DCNN model files (e.g., params.pickle and sd.pt).
    """
    print(f"Attempting to load {n_models} DCNN models from {model_dir}...")
   
    # 1. Determine expected file names (Adjust these based on how you saved them)
    # Assuming models are saved as model_0.pt, model_1.pt, etc.
    params_path = os.path.join(model_dir, "dcnn_params.pickle")
   
    if not os.path.exists(params_path):
        print(f"[ERROR] DCNN parameters file not found at: {params_path}")
        print("Using default initialized models. Predictions will be poor.")
        return
    try:
        # Load hyper-parameters required for DCNN initialization
        with open(params_path, "rb") as f:
            params = pickle.load(f)
       
        # Override the wrapper's default models with the loaded structure
        wrapper.dc_models = []
       
        for i in range(n_models):
            weights_path = os.path.join(model_dir, f"dcnn_model_{i}.pt")
           
            # Create a new DCNN instance using the loaded parameters
            model = DCNN(**params)
           
            if os.path.exists(weights_path):
                # Load the state dictionary (weights)
                state_dict = torch.load(weights_path, map_location=model.device)
                model.load_state_dict(state_dict)
                model.eval()
                wrapper.dc_models.append(model)
                print(f" Successfully loaded DCNN model {i} from {weights_path}")
            else:
                print(f" [WARNING] DCNN model {i} not found at {weights_path}. Using default initialized model.")
                wrapper.dc_models.append(model) # Keep the default if loading fails
       
        print("DCNN ensemble loading complete.")
    except Exception as e:
        print(f"[ERROR] Failed to load DCNN ensemble: {e}")
        print("Proceeding with default initialized models.")
# =============================================================================
# SAC POLICY LOADING FUNCTION
# =============================================================================
def load_sac_policy(path: str, env):
    """Load the trained SAC policy (best_policy.pth or zip)"""
    print(f"Loading trained SAC policy from: {path}")
    policy = SAC.load(path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    print("SAC policy loaded successfully!")
    return policy
# =============================================================================
# 3. ROS 2 PUBLISHER (Unchanged from baseline)
# =============================================================================
class SimPublisher(Node):
    def __init__(self, control_freq: int):
        super().__init__('sim_publisher')
       
        self.pub_leader_q = self.create_publisher(JointState, 'leader/joint_states', 100)
        self.pub_remote_q = self.create_publisher(JointState, 'remote/joint_states', 100)
        self.pub_leader_ee = self.create_publisher(PointStamped, 'leader/ee_pose', 50)
        self.pub_remote_ee = self.create_publisher(PointStamped, 'remote/ee_pose', 50)
        self.pub_obs_delay = self.create_publisher(Float32, 'agent/obs_delay_steps', 50)
        self.pub_act_delay = self.create_publisher(Float32, 'agent/act_delay_steps', 50)
       
        self.control_freq = control_freq
        self.ee_skip = max(1, control_freq // EE_POSE_PUBLISH_FREQ)
        self.metrics_skip = max(1, control_freq // METRICS_PUBLISH_FREQ)
        self.step_count = 0
    def publish_all(self, gt_q, rem_q, leader_ee, remote_ee, obs_delay, act_delay):
        now = self.get_clock().now().to_msg()
        self.step_count += 1
       
        def create_js(q):
            msg = JointState()
            msg.header.stamp = now
            msg.name = [f'panda_joint{i+1}' for i in range(7)]
            msg.position = q.tolist()
            return msg
           
        def create_ps(pos):
            msg = PointStamped()
            msg.header.stamp = now
            msg.header.frame_id = "world"
            msg.point.x, msg.point.y, msg.point.z = float(pos[0]), float(pos[1]), float(pos[2])
            return msg
        self.pub_leader_q.publish(create_js(gt_q))
        self.pub_remote_q.publish(create_js(rem_q))
       
        if self.step_count % self.ee_skip == 0:
            self.pub_leader_ee.publish(create_ps(leader_ee))
            self.pub_remote_ee.publish(create_ps(remote_ee))
       
        if self.step_count % self.metrics_skip == 0:
            obs_msg = Float32()
            obs_msg.data = float(obs_delay)
            self.pub_obs_delay.publish(obs_msg)
           
            act_msg = Float32()
            act_msg.data = float(act_delay)
            self.pub_act_delay.publish(act_msg)
# =============================================================================
# 4. MAIN DEPLOYMENT LOOP (SBSP + RL)
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
   
    # --- Configuration ---
    CONTROL_FREQ = DEFAULT_CONTROL_FREQ
    DT = 1.0 / CONTROL_FREQ
    EXPERIMENT_CONFIG = ExperimentConfig.FULL_RANGE_COVER # Use the same config as the baseline
    DCNN_MODEL_PATH = os.path.join(CHECKPOINT_DIR_RL, "SBSP_DCNN_CHECKPOINT_DIR") # Set this to your saved DCNN dir
    RL_MODEL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent_autoregressive/rl_training_output/SBSP_SAC_FULL_RANGE_COVER_figure_8_20251205_124725/best_policy.pth"
  
    # --- Setup Environment ---
    # 1. Create Base Environment (This contains the Leader and Remote Simulators)
    base_env = TeleoperationEnvWithDelay(
        delay_config=EXPERIMENT_CONFIG,
        # trajectory_type='figure_8',
        randomize_trajectory=False,
        render_mode='human', # We will use the viewer, but keep this for internal logic
        lstm_model_path=None
    )
    # 2. Wrap with SBSP Predictor (This handles the state prediction and observation augmentation)
    rl_env = SBSP_Trajectory_Wrapper(base_env, n_models=5)
   
    # 3. Load Trained DCNN Models into the Wrapper
    load_dcnn_ensemble(rl_env, DCNN_MODEL_PATH, n_models=5)
    # 4. Load Trained RL Policy (SAC)
    # Pass the wrapped environment to handle observation/action spaces correctly
    rl_policy = load_sac_policy(RL_MODEL_PATH, rl_env)
   
    # --- Setup ROS and Internal States ---
    publisher = SimPublisher(CONTROL_FREQ)
   
    # Access the unwrapped components for direct state access/rendering
    unwrapped_env = rl_env.unwrapped
    leader_sim = unwrapped_env.leader_robot
    remote_sim = unwrapped_env.remote_robot
   
    # Initial environment reset
    obs, info = rl_env.reset()
   
    sim_time = 0.0
    steps = 0
   
    print(f"=" * 70)
    print(f"TELEOPERATION DEPLOYMENT - SBSP-RL COMPENSATION")
    print(f"=" * 70)
    print(f" Control Loop: {CONTROL_FREQ} Hz")
    print(f" Mode: Remote tracks SBSP-Predicted Leader State + RL Torque")
    print(f" Total Delay = obs_delay + act_delay (round-trip)")
    print(f"")
    print(f" Timeline:")
    print(f" T=t: Agent observes predicted state $\\hat{s}_t$ from SBSP.")
    print(f" T=t: Agent calculates $\\tau_{{RL}}$ based on $\\hat{s}_t$.")
    print(f" T=t: Remote executes $\\tau_{{PD}}(\\hat{s}_t) + \\tau_{{RL}}$")
    print(f"=" * 70)
   
    # MuJoCo Viewer setup (using the Remote Robot's model)
    with mujoco.viewer.launch_passive(remote_sim.model, remote_sim.data) as viewer:
        while viewer.is_running() and rclpy.ok():
            step_start = time.perf_counter()
           
            # 1. RL Policy Action Selection (Use the latest augmented observation)
            # The action is the 7-DOF torque compensation vector (tau_RL)
            action, _ = rl_policy.predict(obs, deterministic=True)
           
            # 2. Environment Step
            # This single call performs: Leader step, delay simulation, SBSP prediction,
            # Remote PD + RL execution, and updates the observation (obs) for the next step.
            obs, reward, terminated, truncated, info = rl_env.step(action)
           
            # 3. State Extraction for Logging and Publishing
            # True Leader (Ground Truth) state at time t
            gt_q = leader_sim.q
            leader_ee = leader_sim.get_ee_pos()
           
            # Remote state (actual physical state) at time t
            rem_q, rem_qd = remote_sim.get_joint_state()
            remote_ee = remote_sim.get_ee_pos()
           
            # Delay metrics from the environment's info dictionary
            current_obs_delay = info.get('current_delay_steps', 0)
            # The act_delay is constant within a step, so we need to access the simulator for the *next* step's delay
            current_act_delay = unwrapped_env.delay_simulator.get_action_delay_steps()
            total_delay_steps = current_obs_delay + current_act_delay
           
            # 4. Publish & Sync
            publisher.publish_all(gt_q, rem_q, leader_ee, remote_ee,
                                  current_obs_delay, current_act_delay)
           
            steps += 1
            sim_time += DT
            viewer.sync()
           
            # 5. Step Timing (Maintain control frequency)
            elapsed = time.perf_counter() - step_start
            if elapsed < DT:
                time.sleep(DT - elapsed)
               
            # 6. Logging
            if steps % 20 == 0:
                track_err = np.linalg.norm(rem_q - gt_q)
                ee_err = np.linalg.norm(remote_ee - leader_ee)
                total_delay_ms = total_delay_steps * DT * 1000
                pred_err = info.get('prediction_error', 0.0) # From sbsp_wrapper.py
               
                print(f"[Step {steps:5d}] "
                      f"Joint Err: {track_err:.4f} | "
                      f"EE Err: {ee_err:.4f}m | "
                      f"Pred Err: {pred_err:.4f} | "
                      f"Delay: {current_obs_delay}+{current_act_delay}={total_delay_steps} steps ({total_delay_ms:.0f}ms)")
           
            if terminated or truncated:
                print(f"Episode terminated/truncated after {steps} steps.")
                break
    publisher.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main()