"""
Shared Robot configuration setting
"""

import numpy as np
import torch

# Model paths
DEFAULT_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml"

######################################
# Franka panda robot parameters
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

# Initial joint configuration (comfortable home position)
INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

# Joint limits margin to avoid singularities (radians)
JOINT_LIMIT_MARGIN = 0.05 # radians

######################################
# Control parameters
######################################

# Default control frequency (Hz)
DEFAULT_CONTROL_FREQ = 250
CONTROL_CYCLE_TIME = 1.0 / DEFAULT_CONTROL_FREQ # Seconds

# Local robot PD gains (joint-specific)
KP_LOCAL_DEFAULT = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float32)
KD_LOCAL_DEFAULT = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float32)

# Remote robot PD gains (nominal, before adaptation)
KP_REMOTE_NOMINAL = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float32)
KD_REMOTE_NOMINAL = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float32)

######################################
# IK solver parameters
######################################

IK_MAX_ITER = 100
IK_TOLERANCE = 1e-4
IK_DAMPING = 1e-4
IK_STEP_SIZE = 0.25
IK_MAX_JOINT_CHANGE = 0.1
IK_CONTINUITY_GAIN = 0.5

######################################
# Trajectory generation parameters
######################################

# Default trajectory center (meters)
TRAJECTORY_CENTER_DEFAULT = np.array([0.4, 0.0, 0.6], dtype=np.float32)

# Default trajectory scale (meters)
TRAJECTORY_SCALE_DEFAULT = np.array([0.1, 0.3], dtype=np.float32)

# Default frequency (Hz)
TRAJECTORY_FREQUENCY_DEFAULT = 0.1

######################################
# RL agent parameters
######################################

# Maximum steps per episode
MAX_EPISODE_STEPS = 1000

# Termination condition for high joint error (radians)
MAX_JOINT_ERROR_TERMINATION = 3.0  # radians

# Length of history buffers for observation space
ACTION_HISTORY_LEN = 5
TARGET_HISTORY_LEN = 10

MAX_TORQUE_COMPENSATION = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float32)

OBS_DIM = (
    N_JOINTS +  # remote_q_pos (current joint position)
    N_JOINTS +  # remote_q_vel (current joint velocity)
    N_JOINTS +  # delayed_target_q (delayed target joint position)
    (N_JOINTS * TARGET_HISTORY_LEN) +  # target_q_history
    (N_JOINTS * TARGET_HISTORY_LEN) +  # target_qd_history
    1 +         # delay_magnitude
    (N_JOINTS * ACTION_HISTORY_LEN)     # action_history
)

######################################
# Shared model architecture parameters
######################################
# State prediction buffer length
STATE_BUFFER_LENGTH = 20

# RNN architecture for state prediction
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 2
RNN_SEQUENCE_LENGTH = STATE_BUFFER_LENGTH # Must match buffer length

# PPO policy (Actor-Critic) architecture
PPO_MLP_HIDDEN_DIMS = [512, 256]
PPO_ACTIVATION = 'relu'

######################################
# Recurrent-PPO Training Hyperparameters
######################################
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PREDICTION_LOSS_WEIGHT = 5.0
PPO_LOSS_WEIGHT = 1.0
PPO_ROLLOUT_STEPS = 2048
PPO_NUM_EPOCHS = 10
PPO_BATCH_SIZE = 64
PPO_TOTAL_TIMESTEPS = 2_000_000

# Dense reward function weights
REWARD_PREDICTION_WEIGHT = 5.0
REWARD_TRACKING_WEIGHT = 10.0
REWARD_ERROR_SCALE = 100.0
REWARD_VEL_PREDICTION_WEIGHT_FACTOR = 0.3

######################################
# Logging and Checkpointing
######################################
LOG_FREQ = 10
SAVE_FREQ = 100
CHECKPOINT_DIR = "./rl_training_output/checkpoints/recurrent_ppo"

######################################
# Deployment parameters
######################################
INFERENCE_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_INFERENCE_TIME = 0.75 * CONTROL_CYCLE_TIME

######################################
# Early stopping
######################################
ENABLE_EARLY_STOPPING = True         # Set to True to enable
EARLY_STOPPING_PATIENCE = 20         # How many checks without improvement before stopping (e.g., 20 checks * 10 updates/check = 200 updates)
EARLY_STOPPING_MIN_DELTA = 1.0       # Minimum reward improvement considered significant (adjust based on reward scale)
EARLY_STOPPING_CHECK_FREQ = 10       # Check for improvement every N updates

######################################

# Robot configuration validation
def _validate_config():
    """Validate configuration consistency on import."""
    # Basic Robot Params
    assert JOINT_LIMITS_LOWER.shape == (N_JOINTS,)
    assert JOINT_LIMITS_UPPER.shape == (N_JOINTS,)
    assert TORQUE_LIMITS.shape == (N_JOINTS,)
    assert INITIAL_JOINT_CONFIG.shape == (N_JOINTS,)
    assert MAX_TORQUE_COMPENSATION.shape == (N_JOINTS,)
    assert 0 < JOINT_LIMIT_MARGIN < 0.5
    assert np.all(JOINT_LIMITS_LOWER < JOINT_LIMITS_UPPER)
    assert np.all(INITIAL_JOINT_CONFIG >= JOINT_LIMITS_LOWER) and \
           np.all(INITIAL_JOINT_CONFIG <= JOINT_LIMITS_UPPER)
    assert np.all(TORQUE_LIMITS >= 0)
    assert np.all(MAX_TORQUE_COMPENSATION >= 0)
    assert np.all(MAX_TORQUE_COMPENSATION <= TORQUE_LIMITS) # Safety check

    # Control Params
    assert DEFAULT_CONTROL_FREQ > 0
    assert CONTROL_CYCLE_TIME > 0
    assert KP_REMOTE_NOMINAL.shape == (N_JOINTS,)
    assert KD_REMOTE_NOMINAL.shape == (N_JOINTS,)
    if 'KP_LOCAL_DEFAULT' in globals() and KP_LOCAL_DEFAULT is not None:
        assert KP_LOCAL_DEFAULT.shape == (N_JOINTS,)
        assert KD_LOCAL_DEFAULT.shape == (N_JOINTS,)

    # Environment Params
    assert MAX_EPISODE_STEPS > 0
    assert ACTION_HISTORY_LEN >= 0
    assert TARGET_HISTORY_LEN >= 0
    expected_obs_dim = (N_JOINTS * 3 + 
                       (N_JOINTS * TARGET_HISTORY_LEN) * 2 + 
                       1 + 
                       (N_JOINTS * ACTION_HISTORY_LEN))
    assert OBS_DIM == expected_obs_dim, f"OBS_DIM mismatch: calculated {expected_obs_dim}, defined {OBS_DIM}"

    # Model Architecture
    assert STATE_BUFFER_LENGTH > 0
    assert RNN_HIDDEN_DIM > 0
    assert RNN_NUM_LAYERS > 0
    assert RNN_SEQUENCE_LENGTH == STATE_BUFFER_LENGTH
    assert len(PPO_MLP_HIDDEN_DIMS) > 0 and all(d > 0 for d in PPO_MLP_HIDDEN_DIMS)

    # Training Hyperparameters
    assert PPO_LEARNING_RATE > 0
    assert 0 < PPO_GAMMA <= 1
    assert 0 < PPO_GAE_LAMBDA <= 1
    assert PPO_CLIP_EPSILON > 0
    assert PPO_ROLLOUT_STEPS > 0
    assert PPO_NUM_EPOCHS > 0
    assert PPO_BATCH_SIZE > 0 and PPO_BATCH_SIZE <= PPO_ROLLOUT_STEPS
    assert PREDICTION_LOSS_WEIGHT >= 0
    assert PPO_LOSS_WEIGHT >= 0
    assert REWARD_PREDICTION_WEIGHT >= 0
    assert REWARD_TRACKING_WEIGHT >= 0
    assert REWARD_ERROR_SCALE > 0
    assert PPO_TOTAL_TIMESTEPS > PPO_ROLLOUT_STEPS
    assert 0 < REWARD_VEL_PREDICTION_WEIGHT_FACTOR <= 1.0

    # Deployment
    assert MAX_INFERENCE_TIME < CONTROL_CYCLE_TIME, \
        f"Inference time ({MAX_INFERENCE_TIME*1000:.2f}ms) must be less than control cycle ({CONTROL_CYCLE_TIME*1000:.2f}ms)"

    # Logging/Deployment
    assert LOG_FREQ > 0
    assert SAVE_FREQ > 0
    assert CHECKPOINT_DIR != ""

    # Early Stopping
    if ENABLE_EARLY_STOPPING:
        assert EARLY_STOPPING_PATIENCE > 0
        assert EARLY_STOPPING_MIN_DELTA >= 0
        assert EARLY_STOPPING_CHECK_FREQ > 0

    print("=" * 60)
    print("Configuration Validation")
    print("=" * 60)
    print(" Robot and Training configuration validated successfully")
    print(f"\n Control System:")
    print(f"  • Frequency: {DEFAULT_CONTROL_FREQ} Hz ({CONTROL_CYCLE_TIME*1000:.2f} ms/cycle)")
    print(f"  • Max Inference Time: {MAX_INFERENCE_TIME*1000:.2f} ms")
    print(f"  • Safety Margin: {(CONTROL_CYCLE_TIME - MAX_INFERENCE_TIME)*1000:.2f} ms")
    
    print(f"\n Model Architecture:")
    print(f"  • RNN Hidden Dim: {RNN_HIDDEN_DIM}")
    print(f"  • RNN Num Layers: {RNN_NUM_LAYERS}")
    print(f"  • PPO MLP: {PPO_MLP_HIDDEN_DIMS}")
    print(f"  • Sequence Length: {RNN_SEQUENCE_LENGTH}")
    
    print(f"\n Training Configuration:")
    print(f"  • Algorithm: Recurrent-PPO (End-to-End)")
    print(f"  • Total Timesteps: {PPO_TOTAL_TIMESTEPS:,}")
    print(f"  • Rollout: {PPO_ROLLOUT_STEPS} steps × {PPO_NUM_EPOCHS} epochs")
    print(f"  • Batch Size: {PPO_BATCH_SIZE}")
    print(f"  • Learning Rate: {PPO_LEARNING_RATE}")
    print(f"  • Updates: {PPO_TOTAL_TIMESTEPS // PPO_ROLLOUT_STEPS}")
    
    print(f"\n Reward Configuration:")
    print(f"  • Prediction Weight: {REWARD_PREDICTION_WEIGHT}")
    print(f"  • Tracking Weight: {REWARD_TRACKING_WEIGHT}")
    print(f"  • Error Scale: {REWARD_ERROR_SCALE}")
    print(f"  • Velocity Factor: {REWARD_VEL_PREDICTION_WEIGHT_FACTOR}")
    
    print(f"\n Loss Weights:")
    print(f"  • Prediction Loss: {PREDICTION_LOSS_WEIGHT}")
    print(f"  • PPO Loss: {PPO_LOSS_WEIGHT}")
    
    if ENABLE_EARLY_STOPPING:
        print(f"\n Early Stopping:")
        print(f"  • Enabled: Yes")
        print(f"  • Patience: {EARLY_STOPPING_PATIENCE} checks")
        print(f"  • Min Delta: {EARLY_STOPPING_MIN_DELTA}")
        print(f"  • Check Frequency: every {EARLY_STOPPING_CHECK_FREQ} updates")
    
    print("=" * 60)

# Run validation when config is imported
_validate_config()