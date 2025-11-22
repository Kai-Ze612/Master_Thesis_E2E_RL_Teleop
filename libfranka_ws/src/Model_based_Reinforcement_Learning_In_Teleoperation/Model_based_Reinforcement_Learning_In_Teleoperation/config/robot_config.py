"""
Shared Robot configuration settings
"""

import numpy as np

######################################
# File Paths
######################################
DEFAULT_MUJOCO_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml"
RL_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent/rl_training_output/ModelBasedSAC_LOW_DELAY_figure_8_20251121_195833/best_policy.pth"
LSTM_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent/lstm_training_output/Pretrain_LSTM_LateFusion_LOW_DELAY_20251121_170404/estimator_best.pth"

######################################
# DEFAULT_MUJOCO_MODEL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml"
# LSTM_MODEL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent/lstm_training_output/Pretrain_LSTM_LateFusion_MEDIUM_DELAY_20251121_170420/estimator_best.pth"
######################################
CHECKPOINT_DIR_RL = "./rl_training_output"
CHECKPOINT_DIR_LSTM = "./lstm_training_output"


######################################
# Franka Panda Robot Parameters
######################################
N_JOINTS = 7
EE_BODY_NAME = "panda_hand"
TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float32)
JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)
MAX_TORQUE_COMPENSATION = np.array([20.0, 15.0, 15.0, 10.0, 10.0, 5.0, 5.0], dtype=np.float32)

INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

JOINT_LIMIT_MARGIN = 0.05  # Margin to avoid hitting joint limits

KP_LOCAL = np.array([60.0, 50.0, 50.0, 30.0, 30.0, 15.0, 15.0], dtype=np.float32)
KD_LOCAL = np.array([ 13.0,  12.0,  12.0,  10.5, 10.0, 8.0,  8.0], dtype=np.float32) 

WARM_UP_DURATION = 1  # before starting sending trajectory commands

######################################
# Control Parameters
######################################
DEFAULT_CONTROL_FREQ = 100
DEFAULT_PUBLISH_FREQ = 100

DEFAULT_KP_REMOTE = np.array([60.0, 50.0, 50.0, 30.0, 30.0, 15.0, 15.0], dtype=np.float32)
DEFAULT_KD_REMOTE = np.array([ 13.0,  12.0,  12.0,  10.5, 10.0, 8.0,  8.0], dtype=np.float32) 

######################################
# IK Solver Parameterss
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
TRAJECTORY_CENTER = np.array([0.4, 0.0, 0.5], dtype=np.float32)
TRAJECTORY_SCALE = np.array([0.2, 0.2], dtype=np.float32)
TRAJECTORY_FREQUENCY = 0.1  # Hz

######################################
# pre-trained LSTM hyperparameters
######################################
ESTIMATOR_LEARNING_RATE = 3e-3
ESTIMATOR_BATCH_SIZE = 256
ESTIMATOR_BUFFER_SIZE = 200000
ESTIMATOR_TOTAL_UPDATES = 500000
ESTIMATOR_VAL_STEPS = 5000
ESTIMATOR_VAL_FREQ = 1000
ESTIMATOR_PATIENCE = 30
ESTIMATOR_LR_PATIENCE = 5

RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 3
RNN_SEQUENCE_LENGTH = 100 # Input sequence for LSTM

DELAY_INPUT_NORM_FACTOR = 10.0
TARGET_DELTA_SCALE = 500.0

######################################
# RL Environment Parameters
######################################
MAX_EPISODE_STEPS = 10000
MAX_JOINT_ERROR_TERMINATION = 1.0

REMOTE_HISTORY_LEN = 5

OBS_DIM = (
    N_JOINTS +                              # remote_q: 7
    N_JOINTS +                              # remote_qd: 7
    REMOTE_HISTORY_LEN * (N_JOINTS) +       # remote_q_history: 5 * 7
    REMOTE_HISTORY_LEN * (N_JOINTS) +       # remote_qd_history: 5 * 7
    N_JOINTS +                              # predicted_q : 7
    N_JOINTS +                              # predicted_qd: 7
    N_JOINTS +                              # error_q: 7
    N_JOINTS +                                # error_qd: 7
    1                                        # current_delay: 1
)
# Total dim: 112

######################################
# SAC Hyperparameters
######################################
# SAC MLP architecture
SAC_MLP_HIDDEN_DIMS = [512, 256]
SAC_ACTIVATION = 'relu'

# Learning Rates
SAC_LEARNING_RATE = 3e-4        # LR for Actor and Critic
ALPHA_LEARNING_RATE = 3e-4      # LR for temperature auto-tuning

# SAC Parameters
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_TARGET_ENTROPY = 'auto'

# Action distribution numerical stability
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Training Schedule
SAC_BUFFER_SIZE = 1_000_000     # Max size of replay buffer
SAC_BATCH_SIZE = 256            # batch size of gradient updates

# Training Schedule
SAC_START_STEPS = 20000          # Number of random exploration steps (before learning)
SAC_UPDATES_PER_STEP = 1.0       # Number of SAC updates per env step
SAC_TOTAL_TIMESTEPS = 3_000_000  # Total training timesteps

# Validation and Early Stopping
SAC_VAL_FREQ = 25000
SAC_VAL_EPISODES = 10
SAC_EARLY_STOPPING_PATIENCE = 10

######################################
# Reward Function Configuration
######################################
TRACKING_ERROR_SCALE = 10       # Gaussian bandwidth for exp(-scale * error²)
VELOCITY_ERROR_SCALE = 5       # Gaussian bandwidth for velocity tracking

ACTION_PENALTY_WEIGHT = 0.01 # penalty for large actions

######################################
# Environment Settings
######################################

# NUM_ENVIRONMENTS = 5 # Number of parallel environments for training
NUM_ENVIRONMENTS = 1

######################################
# Logging and Checkpointing
######################################
LOG_FREQ = 100   # Log metrics every N *env steps*
SAVE_FREQ = 1000  # Save checkpoint every N *env steps*

######################################
# Deployment Parameters
######################################

MAX_INFERENCE_TIME = 0.9 * (1.0 / DEFAULT_CONTROL_FREQ)  # 90% of control cycle time for safety

DEPLOYMENT_HISTORY_BUFFER_SIZE = 200  # Must be > max_delay_steps + RNN sequence length

