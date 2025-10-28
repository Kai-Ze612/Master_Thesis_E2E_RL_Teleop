"""
Shared Robot configuration settings
"""

import numpy as np
import torch

######################################
# File Paths
######################################

# Model paths
DEFAULT_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml"

# RL model paths
DEFAULT_RL_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Reinforcement_Learning_In_Teleoperation/Reinforcement_Learning_In_Teleoperation/rl_agent/rl_training_output"

######################################
# Franka Panda Robot Parameters
######################################

N_JOINTS = 7
EE_BODY_NAME = "panda_hand"
TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float32)  # meters

# Joint limits (radians)
JOINT_LIMITS_LOWER = np.array([
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
], dtype=np.float32)

JOINT_LIMITS_UPPER = np.array([
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
], dtype=np.float32)

# Torque limits (Nm)
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)

# Maximum torque compensation for RL agent (Nm)
MAX_TORQUE_COMPENSATION = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float32)

# Initial joint configuration (comfortable home position)
INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

# Joint limits margin to avoid singularities (radians)
JOINT_LIMIT_MARGIN = 0.05  # radians

######################################
# Control Parameters
######################################

# Default control frequency (Hz)
DEFAULT_CONTROL_FREQ = 500 # match the sim freq

# Default publish frequency for robot state (Hz)
DEFAULT_PUBLISH_FREQ = 100

# Local robot PD gains (joint-specific)
KP_LOCAL_DEFAULT = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float32)
KD_LOCAL_DEFAULT = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float32)

# Remote robot PD gains, setting softer because of stable under delay
KP_REMOTE_DEFAULT = np.array([150.0, 150.0, 150.0, 150.0, 60.0, 60.0, 15.0], dtype=np.float32)
KD_REMOTE_DEFAULT = np.array([30.0, 30.0, 30.0, 30.0, 20.0, 20.0, 5.0], dtype=np.float32)

######################################
# IK Solver Parameters
######################################

IK_MAX_ITER = 100
IK_TOLERANCE = 1e-4
IK_DAMPING = 1e-4
IK_STEP_SIZE = 0.25
IK_MAX_JOINT_CHANGE = 0.1
IK_CONTINUITY_GAIN = 0.5

######################################
# Trajectory Generation Parameters
######################################

# Default trajectory center (meters)
TRAJECTORY_CENTER_DEFAULT = np.array([0.4, 0.0, 0.6], dtype=np.float32)

# Default trajectory scale (meters)
TRAJECTORY_SCALE_DEFAULT = np.array([0.1, 0.3], dtype=np.float32)

# Default frequency (Hz)
TRAJECTORY_FREQUENCY_DEFAULT = 0.1

######################################
# RL Environment Parameters
######################################

# Maximum steps per episode
MAX_EPISODE_STEPS = 2000

# Termination condition for high joint error (radians)
MAX_JOINT_ERROR_TERMINATION = 3.0  # radians

# Length of history buffers for observation space
TARGET_HISTORY_LEN = 5  # How many leader trajectory points to keep
ACTION_HISTORY_LEN = 5   # How many past actions to buffer

OBS_DIM = (
    N_JOINTS +                              # remote_q: 7
    N_JOINTS +                              # remote_qd: 7
    N_JOINTS +                              # predicted_q: 7
    N_JOINTS +                              # predicted_qd: 7
    (N_JOINTS * TARGET_HISTORY_LEN) +       # target_q_history: 35
    (N_JOINTS * TARGET_HISTORY_LEN) +       # target_qd_history: 35
    1 +                                     # delay_magnitude: 1
    (N_JOINTS * ACTION_HISTORY_LEN)         # action_history: 35
)  # Total: 134 dimensions

######################################
# Model Architecture Parameters
######################################

# State prediction buffer length
STATE_BUFFER_LENGTH = 256

# RNN (LSTM) architecture for state prediction
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 2
RNN_SEQUENCE_LENGTH = STATE_BUFFER_LENGTH  # Must match buffer length

# PPO policy (Actor-Critic) architecture
PPO_MLP_HIDDEN_DIMS = [512, 256]
PPO_ACTIVATION = 'relu'

######################################
# Recurrent-PPO Training Hyperparameters
######################################

# Learning
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.9
PPO_GAE_LAMBDA = 0.95

# PPO-specific
PPO_CLIP_EPSILON = 0.2
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

# Loss weighting
PREDICTION_LOSS_WEIGHT = 5.0  # Weight for supervised state prediction loss
PPO_LOSS_WEIGHT = 1.0          # Weight for PPO loss (actor + critic + entropy)

# Training schedule
PPO_ROLLOUT_STEPS = 5000
PPO_NUM_EPOCHS = 10
PPO_BATCH_SIZE = 128
PPO_TOTAL_TIMESTEPS = 2_000_000

######################################
# Dense Reward Function Weights
######################################

# Reward component weights
REWARD_PREDICTION_WEIGHT = 5.0   # Weight for state prediction accuracy
REWARD_TRACKING_WEIGHT = 10.0    # Weight for tracking performance

# Reward scaling
REWARD_ERROR_SCALE = 100.0       # Scale factor for exponential reward
REWARD_VEL_PREDICTION_WEIGHT_FACTOR = 0.3  # Weight for velocity prediction vs position

NUM_ENVIRONMENTS = 8   # Number of parallel environments
######################################
# Logging and Checkpointing
######################################

LOG_FREQ = 10   # Log metrics every N updates
SAVE_FREQ = 100  # Save checkpoint every N updates
CHECKPOINT_DIR = "./rl_training_output/recurrent_ppo"

######################################
# Deployment Parameters
######################################

MAX_INFERENCE_TIME = 0.9 * (1.0 / DEFAULT_CONTROL_FREQ)  # 90% of control cycle time for safety

######################################
# Early Stopping Configuration
######################################

ENABLE_EARLY_STOPPING = True           # Set to True to enable early stopping
EARLY_STOPPING_PATIENCE = 10           # Number of checks without improvement before stopping
EARLY_STOPPING_MIN_DELTA = 1.0         # Minimum reward improvement to be considered significant
EARLY_STOPPING_CHECK_FREQ = 10         # Check for improvement every N updates

######################################
# Configuration Validation
######################################

def _print_config():
    print("Robot Configuration for Franka Panda:")
    print(f"N_JOINTS: {N_JOINTS}")
    print(f"DEFAULT_CONTROL_FREQ: {DEFAULT_CONTROL_FREQ} Hz")
    print(f"KP_REMOTE_DEFAULT: {KP_REMOTE_DEFAULT}")
    print(f"KD_REMOTE_DEFAULT: {KD_REMOTE_DEFAULT}")
    print(f"OBS_DIM: {OBS_DIM}")
    print("*" * 70)
    print(f"TAJECTORY_CENTER_DEFAULT: {TRAJECTORY_CENTER_DEFAULT}")
    print(f"TRAJECTORY_SCALE_DEFAULT: {TRAJECTORY_SCALE_DEFAULT}")
    print("*" * 70)
    print(f"TARGET_HISTORY_LEN: {TARGET_HISTORY_LEN}")
    print(f"ACTION_HISTORY_LEN: {ACTION_HISTORY_LEN}")
    print("*" * 70)
    print(f"STATE_BUFFER_LENGTH: {STATE_BUFFER_LENGTH}")
    print(f"RNN_HIDDEN_DIM: {RNN_HIDDEN_DIM}")
    print(f"RNN_NUM_LAYERS: {RNN_NUM_LAYERS}")
    print(f"RNN_SEQUENCE_LENGTH: {RNN_SEQUENCE_LENGTH}")
    print(f"PPO_LEARNING_RATE: {PPO_LEARNING_RATE}")
    print(f"PPO_TOTAL_TIMESTEPS: {PPO_TOTAL_TIMESTEPS}")
    print("*" * 70)
    print(f"NUM_ENVIRONMENTS: {NUM_ENVIRONMENTS}")


_print_config()