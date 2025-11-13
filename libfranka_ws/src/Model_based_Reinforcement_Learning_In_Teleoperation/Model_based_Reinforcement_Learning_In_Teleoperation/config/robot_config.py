"""
Shared Robot configuration settings
"""

import numpy as np

######################################
# File Paths
# (Unchanged)
######################################
DEFAULT_MUJOCO_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml"
DEFAULT_RL_MODEL_PATH_BASE = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Reinforcement_Learning_In_Teleoperation/Reinforcement_Learning_In_Teleoperation/rl_agent/rl_training_output"
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
MAX_TORQUE_COMPENSATION = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float32)
INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
JOINT_LIMIT_MARGIN = 0.05
KP_LOCAL = np.array([150.0, 150.0, 120.0, 120.0, 75.0, 50.0, 20.0], dtype=np.float32)
KD_LOCAL = np.array([20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 5.0], dtype=np.float32)
WARM_UP_DURATION = 1.0

######################################
# Control Parameters
######################################
DEFAULT_CONTROL_FREQ = 500
DEFAULT_PUBLISH_FREQ = 500
DEFAULT_KP_REMOTE = np.array([150.0, 130.0, 100.0, 100.0, 100.0, 30.0, 10.0], dtype=np.float32)
DEFAULT_KD_REMOTE = np.array([15.0, 15.0, 15.0, 10.0, 5.0, 5.0, 2.0], dtype=np.float32)

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
TRAJECTORY_CENTER = np.array([0.4, 0.0, 0.6], dtype=np.float32)
TRAJECTORY_SCALE = np.array([0.1, 0.3], dtype=np.float32)
TRAJECTORY_FREQUENCY = 0.1

######################################
# pre-trained LSTM hyperparameters - OPTIMIZED
######################################
ESTIMATOR_LEARNING_RATE = 1e-3  # Reduced from 3e-3 for more stable training
ESTIMATOR_BATCH_SIZE = 128      # Reduced from 256 - better for smaller sequence length
ESTIMATOR_BUFFER_SIZE = 100000  # Reduced from 200k - sufficient for pre-training
ESTIMATOR_WARMUP_STEPS = 2000   # Reduced from 5000 - faster warmup with smaller seq len
ESTIMATOR_TOTAL_UPDATES = 30000 # Reduced from 50k - faster pre-training
ESTIMATOR_VAL_STEPS = 2000      # Reduced proportionally
ESTIMATOR_VAL_FREQ = 500        # More frequent validation
ESTIMATOR_PATIENCE = 15         # Increased patience for better convergence
ESTIMATOR_LR_PATIENCE = 5

######################################
# RL Environment Parameters
######################################
MAX_EPISODE_STEPS = 1000
MAX_JOINT_ERROR_TERMINATION = 1.0
TARGET_HISTORY_LEN = 5
ACTION_HISTORY_LEN = 5
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
# OPTIMIZED: Reduced sequence length for faster, more stable training
STATE_BUFFER_LENGTH = 64 # Reduced from 256 - still captures ~128ms of history at 500Hz
RNN_HIDDEN_DIM = 128      # Reduced from 256 - sufficient for 7-DOF robot
RNN_NUM_LAYERS = 2        # Reduced from 4 - simpler is often better
RNN_SEQUENCE_LENGTH = STATE_BUFFER_LENGTH # Input sequence for LSTM

# SAC MLP architecture
SAC_MLP_HIDDEN_DIMS = [256, 256]  # Simplified from [512, 256]
SAC_ACTIVATION = 'relu'

######################################
# Model-Based SAC Training Hyperparameters
######################################

# Learning Rates - OPTIMIZED
SAC_LEARNING_RATE = 3e-4        # LR for Actor and Critic
ESTIMATOR_LEARNING_RATE = 3e-4  # Increased from 1e-4 for faster LSTM training
ALPHA_LEARNING_RATE = 3e-4      # LR for temperature auto-tuning

# SAC Parameters
SAC_GAMMA = 0.99                # Discount factor
SAC_TAU = 0.005                 # Polyak averaging coefficient
SAC_TARGET_ENTROPY = 'auto'     # Target entropy for temperature tuning


# Action distribution numerical stability
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Training Schedule - OPTIMIZED
SAC_BUFFER_SIZE = 100_000       # Reduced from 1M - more suitable for robot control
SAC_BATCH_SIZE = 256            # Minibatch size for updates
SAC_START_STEPS = 2500          # Reduced from 5000 - start learning earlier
SAC_UPDATES_PER_STEP = 1.0      # Number of updates per env step (1.0 means 1 update per step)
PPO_TOTAL_TIMESTEPS = 1_000_000 # Reduced from 3M for faster initial testing

# Loss Weighting
# Weight for supervised state prediction loss (applied in SACTrainer)
PREDICTION_LOSS_WEIGHT = 5.0    

######################################
# Dense Reward Function Weights - OPTIMIZED
######################################
REWARD_PREDICTION_WEIGHT = 0.0  # Disabled - prediction is not part of RL reward
REWARD_TRACKING_WEIGHT = 1.0    # Main reward signal
REWARD_TRACKING_SCALE = 0.2
REWARD_ACTION_PENALTY = 0.001   # Reduced from 0.01 - less aggressive penalty
REWARD_ERROR_SCALE_HIGH_ERROR = 5   # Reduced for smoother gradients
REWARD_ERROR_SCALE_MID_ERROR = 15   # Reduced for smoother gradients
REWARD_ERROR_SCALE_LOW_ERROR = 30   # Reduced from 50 for smoother gradients
REWARD_VEL_PREDICTION_WEIGHT_FACTOR = 1.0  # Reduced from 1.5 - equal weighting

NUM_ENVIRONMENTS = 4  # Reduced from 5 for more stable training

######################################
# Logging and Checkpointing
######################################
LOG_FREQ = 100   # Log metrics every N *env steps*
SAVE_FREQ = 1000  # Save checkpoint every N *env steps*