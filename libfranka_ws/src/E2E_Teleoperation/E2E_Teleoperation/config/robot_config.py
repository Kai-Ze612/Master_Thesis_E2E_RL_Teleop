"""
Shared Robot configuration settings
"""

import numpy as np
from pathlib import Path

######################################
# File Paths
######################################
CONFIG_FILE_PATH = Path(__file__).resolve()  # current file path

CONFIG_DIR = CONFIG_FILE_PATH.parent  # Config foler

PACKAGE_ROOT = CONFIG_DIR.parent # Package folder 

PROJECT_ROOT = PACKAGE_ROOT.parent # Project folder

WORKSPACE_SRC = PROJECT_ROOT.parent # Workspace src folder

# RL training model checkpoint
CHECKPOINT_DIR = PACKAGE_ROOT / "trained_RL"
LOG_DIR = PACKAGE_ROOT / "logs"

# Mujoco path
DEFAULT_MUJOCO_MODEL_PATH = (
    WORKSPACE_SRC / 
    "multipanda_ros2" / 
    "franka_description" / 
    "mujoco" / 
    "franka" / 
    "scene.xml"
)

DEFAULT_RL_MODEL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/E2E_Teleoperation/E2E_Teleoperation/trained_RL/E2E_RL_HIGH_VARIANCE_figure_8_20251215_210023/best_policy.pth"

# Ensure the path exist
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
JOINT_LIMIT_MARGIN = 0.05  # Margin to avoid hitting joint limits

# Physical torque limits
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)
MAX_ACTION_TORQUE = TORQUE_LIMITS.copy()

INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)
WARM_UP_DURATION = 1  # sec (before starting moving)
NO_DELAY_DURATION = 1  # sec (before starting delay simulation)

# Normalization Statistics
Q_MEAN = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78], dtype=np.float32)
Q_STD  = np.array([1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32) 
QD_MEAN = np.zeros(7, dtype=np.float32)
QD_STD  = np.ones(7, dtype=np.float32) * 2.0

DELAY_INPUT_NORM_FACTOR = 100.0  # Normalize delay time

######################################
# Simulation and Control Parameters
######################################
CONTROL_FREQ = 250
DT = 1.0 / CONTROL_FREQ
WARM_UP_DURATION = 0.1
NO_DELAY_DURATION = 0.1
MAX_EPISODE_STEPS = 5500
MAX_JOINT_ERROR_TERMINATION = 1.0

######################################
# IK Solver Parameterss
######################################
# 1. Tolerances & Iterations
IK_POSITION_TOLERANCE = 0.01 # meters
IK_JACOBIAN_MAX_ITER = 200
IK_OPTIMIZATION_MAX_ITER = 100

# 2. Damping & Step
IK_JACOBIAN_STEP_SIZE = 0.01 # Larger = Faster but less stable
IK_JACOBIAN_DAMPING = 0.1    # Larger = More stable but less responsive

IK_NULL_SPACE_GAIN = 0.5

######################################
# Trajectory Generation Parameters
######################################
TRAJECTORY_CENTER = np.array([0.3, 0, 0.5], dtype=np.float32)
TRAJECTORY_SCALE = np.array([0.2, 0.2, 0.02], dtype=np.float32)
TRAJECTORY_FREQUENCY = 0.1  # Hz

######################################
# Network Architecture
######################################
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 3
RNN_SEQUENCE_LENGTH = 80
ESTIMATOR_STATE_DIM = 15
ESTIMATOR_OUTPUT_DIM = 14
MLP_HIDDEN_DIMS = [512, 256]
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Observation Space
ROBOT_STATE_DIM = 14
ROBOT_HISTORY_DIM = RNN_SEQUENCE_LENGTH * ROBOT_STATE_DIM
TARGET_HISTORY_DIM = RNN_SEQUENCE_LENGTH * ESTIMATOR_STATE_DIM
OBS_DIM = ROBOT_STATE_DIM + ROBOT_HISTORY_DIM + TARGET_HISTORY_DIM

# --- Training Stages ---
SEED = 42
BATCH_SIZE = 2048
BUFFER_SIZE = 1_000_000

STAGE1_STEPS = 10_000
ENCODER_LR = 1e-3

STAGE2_TOTAL_STEPS = 1_000_000
STAGE2A_TOTAL_STEPS = 100_000
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
ALPHA_LR = 1e-4 
GAMMA = 0.99
TAU = 0.005

STAGE3_TOTAL_STEPS = 1_000_000
CRITIC_LR = 3e-4
TARGET_ENTROPY = -7.0    # Usually -Action_Dim

MAX_GRAD_NORM = 1.0
POLICY_DELAY = 2
VAL_FREQ = 10_000
VAL_EPISODES = 5
EARLY_STOP_PATIENCE = 10
CHECKPOINT_FREQ = 50_000

LOG_FREQ = 1_000

######################################
# Teacher Model Gains
######################################
TEACHER_KP = np.array([100.0, 100.0, 100.0, 100.0, 80.0, 60.0, 40.0], dtype=np.float64)
TEACHER_KD = np.array([20.0, 20.0, 20.0, 20.0, 12.0, 10.0, 8.0], dtype=np.float64)
TEACHER_SMOOTHING = 0.3
