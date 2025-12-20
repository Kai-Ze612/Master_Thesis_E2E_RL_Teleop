"""
Shared Robot configuration settings
"""

import numpy as np
from pathlib import Path

######################################
# File Paths
######################################
CONFIG_FILE_PATH = Path(__file__).resolve()
CONFIG_DIR = CONFIG_FILE_PATH.parent
PACKAGE_ROOT = CONFIG_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_SRC = PROJECT_ROOT.parent

CHECKPOINT_DIR = PACKAGE_ROOT / "trained_RL"
LOG_DIR = PACKAGE_ROOT / "logs"

DEFAULT_MUJOCO_MODEL_PATH = (
    WORKSPACE_SRC / 
    "multipanda_ros2" / 
    "franka_description" / 
    "mujoco" / 
    "franka" / 
    "scene.xml"
)

# Ensure paths exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

######################################
# Franka Panda Robot Parameters
######################################
N_JOINTS = 7
EE_BODY_NAME = "panda_hand"
TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float32)

# Robot joints physical limits
JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.5445, -3.0159], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.5169, 3.0159], dtype=np.float32)
JOINT_LIMIT_MARGIN = 0.05

# Physical torque limits
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)
MAX_ACTION_TORQUE = TORQUE_LIMITS.copy()

INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)

# Normalization Statistics
Q_MEAN = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78], dtype=np.float32)
Q_STD  = np.array([1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32) 
QD_MEAN = np.zeros(7, dtype=np.float32)
QD_STD  = np.ones(7, dtype=np.float32) * 2.0

DELAY_INPUT_NORM_FACTOR = 100.0

######################################
# Simulation and Control Parameters
######################################
CONTROL_FREQ = 250
DT = 1.0 / CONTROL_FREQ
WARM_UP_DURATION = 0.1
NO_DELAY_DURATION = 0.1

MAX_EPISODE_STEPS = 5500  # Max episode length, after reaching, the environment resets

MAX_JOINT_ERROR_TERMINATION = 1.0  # After this, the episode terminates

BUFFER_SIZE = 1_000_000

SEED = 42

######################################
# Trajectory Generation Parameters
######################################
TRAJECTORY_CENTER = np.array([0.3, 0, 0.5], dtype=np.float32)
TRAJECTORY_SCALE = np.array([0.2, 0.2, 0.02], dtype=np.float32)
TRAJECTORY_FREQUENCY = 0.1  # Hz

# Randomization Bounds (for LocalRobotSimulator)
TRAJ_RANDOM_CENTER_X = (0.3, 0.4)
TRAJ_RANDOM_CENTER_Y = (-0.1, 0.1)

TRAJ_RANDOM_SCALE_X  = (0.1, 0.1) # Fixed
TRAJ_RANDOM_SCALE_Y  = (0.1, 0.3)

TRAJ_RANDOM_FREQ     = (0.05, 0.15)

######################################
# Network Architecture
######################################
# LSTM Encoder
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 3
RNN_SEQUENCE_LENGTH = 80
ESTIMATOR_STATE_DIM = 15
ESTIMATOR_OUTPUT_DIM = 14
LSTM_PRED_HEAD_DIM = 128     # Hidden layer for prediction head
LSTM_AR_PROJ_DIM = 64        # Hidden layer for AR projection

# Actor/Critic MLP
MLP_HIDDEN_DIMS = [512, 256]
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Observation Dimensions
ROBOT_STATE_DIM = 14
ROBOT_HISTORY_DIM = RNN_SEQUENCE_LENGTH * ROBOT_STATE_DIM
TARGET_HISTORY_DIM = RNN_SEQUENCE_LENGTH * ESTIMATOR_STATE_DIM
OBS_DIM = ROBOT_STATE_DIM + ROBOT_HISTORY_DIM + TARGET_HISTORY_DIM

######################################
# Training Hyperparameters
######################################
BATCH_SIZE = 2048
GAMMA = 0.99
POLYAK_TAU = 0.005
MAX_GRAD_NORM = 1.0

# General Training Settings
VAL_FREQ = 10_000
VAL_EPISODES = 5
EARLY_STOP_PATIENCE = 10
CHECKPOINT_FREQ = 50_000
LOG_INTERVAL = 1000

# --- STAGE 1: Encoder Pre-training ---
STAGE1_STEPS = 10_000 #
STAGE1_COLLECTION_STEPS = 5000
ENCODER_LR = 1e-3

# --- STAGE 2: Behavioral Cloning & DAgger ---
STAGE2A_TOTAL_STEPS = 100_000
STAGE2_TOTAL_STEPS = 1_000_000
STAGE2_COLLECTION_STEPS = 5000
STAGE2_RECOVERY_WEIGHT = 1.5
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
ALPHA_LR = 1e-4 

# Noise Schedule: [(step_threshold, noise_scale_nm)]
STAGE2_NOISE_SCHEDULE = [
    (0, 0.5), 
    (10000, 1.0), 
    (20000, 2.0), 
    (30000, 3.0), 
    (40000, 4.0)
]

# --- STAGE 3: SAC Fine-tuning ---
STAGE3_TOTAL_STEPS = 1_000_000
TARGET_ENTROPY = -7.0

# Stage 3 Specific Weights and Schedules
S3_CRITIC_WARMUP_STEPS = 20000
S3_BC_DECAY_STEPS = 200000
S3_INITIAL_BC_WEIGHT = 10.0
S3_MIN_BC_WEIGHT = 2.5
S3_REWARD_SCALE = 0.01

# Stage 3 Learning Rate Scaling (vs Base LR)
S3_LR_SCALE = 0.05
S3_ALPHA_LR_SCALE = 0.05

######################################
# Teacher Model Gains
######################################
TEACHER_KP = np.array([300.0, 300.0, 300.0, 300.0, 150.0, 150.0, 40.0], dtype=np.float64)
TEACHER_KD = np.array([30.0, 30.0, 30.0, 30.0, 20.0, 20.0, 8.0], dtype=np.float64)
TEACHER_SMOOTHING = 0.3

# Reward Function Parameters
REWARD_DISTANCE_SCALE = 5.0  # exp(-5.0 * dist)

######################################
# Inverse Kinematics (IK)
######################################
IK_POSITION_TOLERANCE = 0.01
IK_JACOBIAN_MAX_ITER = 200
IK_OPTIMIZATION_MAX_ITER = 100
IK_JACOBIAN_STEP_SIZE = 0.01
IK_JACOBIAN_DAMPING = 0.1
IK_NULL_SPACE_GAIN = 0.5

DEFAULT_RL_MODEL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/E2E_Teleoperation/E2E_Teleoperation/trained_RL/E2E_RL_HIGH_VARIANCE_figure_8_20251220_173947/best_policy_stage3.pth"