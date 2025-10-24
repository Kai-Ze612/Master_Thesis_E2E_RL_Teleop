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
DEFAULT_CONTROL_FREQ = 250
CONTROL_CYCLE_TIME = 1.0 / DEFAULT_CONTROL_FREQ  # Seconds

# Default publish frequency for robot state (Hz)
DEFAULT_PUBLISH_FREQ = 100

# Local robot PD gains (joint-specific)
KP_LOCAL_DEFAULT = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float32)
KD_LOCAL_DEFAULT = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float32)

# Remote robot PD gains (nominal, before adaptation)
KP_REMOTE_NOMINAL = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float32)
KD_REMOTE_NOMINAL = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float32)

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
MAX_EPISODE_STEPS = 1000

# Termination condition for high joint error (radians)
MAX_JOINT_ERROR_TERMINATION = 3.0  # radians

# Length of history buffers for observation space
TARGET_HISTORY_LEN = 10  # How many leader trajectory points to keep
ACTION_HISTORY_LEN = 5   # How many past actions to buffer

OBS_DIM = (
    N_JOINTS +                              # remote_q: 7
    N_JOINTS +                              # remote_qd: 7
    N_JOINTS +                              # delayed_target_q: 7
    (N_JOINTS * TARGET_HISTORY_LEN) +       # target_q_history: 70
    (N_JOINTS * TARGET_HISTORY_LEN) +       # target_qd_history: 70
    1 +                                     # delay_magnitude: 1
    (N_JOINTS * ACTION_HISTORY_LEN)         # action_history: 35
)  # Total: 197 dimensions

######################################
# Model Architecture Parameters
######################################

# State prediction buffer length
STATE_BUFFER_LENGTH = 20

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
PPO_GAMMA = 0.99
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
PPO_ROLLOUT_STEPS = 2048
PPO_NUM_EPOCHS = 10
PPO_BATCH_SIZE = 64
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

######################################
# Logging and Checkpointing
######################################

LOG_FREQ = 10   # Log metrics every N updates
SAVE_FREQ = 100  # Save checkpoint every N updates
CHECKPOINT_DIR = "./rl_training_output/checkpoints/recurrent_ppo"

######################################
# Deployment Parameters
######################################

INFERENCE_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_INFERENCE_TIME = 0.9 * CONTROL_CYCLE_TIME  # 90% of control cycle for safety

######################################
# Early Stopping Configuration
######################################

ENABLE_EARLY_STOPPING = True           # Set to True to enable early stopping
EARLY_STOPPING_PATIENCE = 20           # Number of checks without improvement before stopping
EARLY_STOPPING_MIN_DELTA = 1.0         # Minimum reward improvement to be considered significant
EARLY_STOPPING_CHECK_FREQ = 10         # Check for improvement every N updates

######################################
# Configuration Validation
######################################

def _validate_config():
    """
    Validate configuration consistency on import.
    
    This function checks all configuration parameters for:
    - Correct shapes and dimensions
    - Valid ranges and constraints
    - Internal consistency
    """
    
    # ============================================================
    # Basic Robot Parameters
    # ============================================================
    assert JOINT_LIMITS_LOWER.shape == (N_JOINTS,), \
        f"JOINT_LIMITS_LOWER must have shape ({N_JOINTS},)"
    assert JOINT_LIMITS_UPPER.shape == (N_JOINTS,), \
        f"JOINT_LIMITS_UPPER must have shape ({N_JOINTS},)"
    assert TORQUE_LIMITS.shape == (N_JOINTS,), \
        f"TORQUE_LIMITS must have shape ({N_JOINTS},)"
    assert MAX_TORQUE_COMPENSATION.shape == (N_JOINTS,), \
        f"MAX_TORQUE_COMPENSATION must have shape ({N_JOINTS},)"
    assert INITIAL_JOINT_CONFIG.shape == (N_JOINTS,), \
        f"INITIAL_JOINT_CONFIG must have shape ({N_JOINTS},)"
    
    assert 0 < JOINT_LIMIT_MARGIN < 0.5, \
        "JOINT_LIMIT_MARGIN should be reasonable (0 < margin < 0.5 rad)"
    assert np.all(JOINT_LIMITS_LOWER < JOINT_LIMITS_UPPER), \
        "All lower joint limits must be strictly less than upper joint limits"
    assert np.all(INITIAL_JOINT_CONFIG >= JOINT_LIMITS_LOWER) and \
           np.all(INITIAL_JOINT_CONFIG <= JOINT_LIMITS_UPPER), \
           "Initial joint configuration must be within joint limits"
    assert np.all(TORQUE_LIMITS > 0), \
        "Torque limits must be positive"
    assert np.all(MAX_TORQUE_COMPENSATION > 0), \
        "Max torque compensation must be positive"
    assert np.all(MAX_TORQUE_COMPENSATION <= TORQUE_LIMITS), \
        "Max torque compensation must not exceed torque limits"
    
    # ============================================================
    # Control Parameters
    # ============================================================
    assert DEFAULT_CONTROL_FREQ > 0, "Control frequency must be positive"
    assert CONTROL_CYCLE_TIME > 0, "Control cycle time must be positive"
    assert KP_LOCAL_DEFAULT.shape == (N_JOINTS,), \
        f"KP_LOCAL_DEFAULT must have shape ({N_JOINTS},)"
    assert KD_LOCAL_DEFAULT.shape == (N_JOINTS,), \
        f"KD_LOCAL_DEFAULT must have shape ({N_JOINTS},)"
    assert KP_REMOTE_NOMINAL.shape == (N_JOINTS,), \
        f"KP_REMOTE_NOMINAL must have shape ({N_JOINTS},)"
    assert KD_REMOTE_NOMINAL.shape == (N_JOINTS,), \
        f"KD_REMOTE_NOMINAL must have shape ({N_JOINTS},)"
    assert np.all(KP_LOCAL_DEFAULT >= 0), "Local Kp gains must be non-negative"
    assert np.all(KD_LOCAL_DEFAULT >= 0), "Local Kd gains must be non-negative"
    assert np.all(KP_REMOTE_NOMINAL >= 0), "Remote Kp gains must be non-negative"
    assert np.all(KD_REMOTE_NOMINAL >= 0), "Remote Kd gains must be non-negative"
    
    # ============================================================
    # Trajectory Parameters
    # ============================================================
    assert TCP_OFFSET.shape == (3,), "TCP_OFFSET must be a 3D vector"
    assert TRAJECTORY_CENTER_DEFAULT.shape == (3,), \
        "TRAJECTORY_CENTER_DEFAULT must be a 3D vector"
    
    # ============================================================
    # Environment Parameters
    # ============================================================
    assert MAX_EPISODE_STEPS > 0, "MAX_EPISODE_STEPS must be positive"
    assert MAX_JOINT_ERROR_TERMINATION > 0, \
        "MAX_JOINT_ERROR_TERMINATION must be positive"
    assert TARGET_HISTORY_LEN > 0, "TARGET_HISTORY_LEN must be positive"
    assert ACTION_HISTORY_LEN > 0, "ACTION_HISTORY_LEN must be positive"
    
    # Validate OBS_DIM calculation
    expected_obs_dim = (
        N_JOINTS +                              # remote_q
        N_JOINTS +                              # remote_qd
        N_JOINTS +                              # delayed_target_q
        (N_JOINTS * TARGET_HISTORY_LEN) +       # target_q_history
        (N_JOINTS * TARGET_HISTORY_LEN) +       # target_qd_history
        1 +                                     # delay_magnitude
        (N_JOINTS * ACTION_HISTORY_LEN)         # action_history
    )
    assert OBS_DIM == expected_obs_dim, \
        f"OBS_DIM mismatch: calculated {expected_obs_dim}, defined {OBS_DIM}"
    
    # ============================================================
    # Model Architecture
    # ============================================================
    assert STATE_BUFFER_LENGTH > 0, "STATE_BUFFER_LENGTH must be positive"
    assert RNN_HIDDEN_DIM > 0, "RNN_HIDDEN_DIM must be positive"
    assert RNN_NUM_LAYERS > 0, "RNN_NUM_LAYERS must be positive"
    assert RNN_SEQUENCE_LENGTH == STATE_BUFFER_LENGTH, \
        "RNN_SEQUENCE_LENGTH must match STATE_BUFFER_LENGTH"
    assert len(PPO_MLP_HIDDEN_DIMS) > 0, "PPO_MLP_HIDDEN_DIMS must not be empty"
    assert all(d > 0 for d in PPO_MLP_HIDDEN_DIMS), \
        "All PPO_MLP_HIDDEN_DIMS values must be positive"
    
    # ============================================================
    # Training Hyperparameters
    # ============================================================
    assert PPO_LEARNING_RATE > 0, "Learning rate must be positive"
    assert 0 < PPO_GAMMA <= 1, "Gamma must be in (0, 1]"
    assert 0 < PPO_GAE_LAMBDA <= 1, "GAE lambda must be in (0, 1]"
    assert PPO_CLIP_EPSILON > 0, "Clip epsilon must be positive"
    assert PPO_ENTROPY_COEF >= 0, "Entropy coefficient must be non-negative"
    assert PPO_VALUE_COEF >= 0, "Value coefficient must be non-negative"
    assert PPO_MAX_GRAD_NORM > 0, "Max grad norm must be positive"
    
    assert PPO_ROLLOUT_STEPS > 0, "Rollout steps must be positive"
    assert PPO_NUM_EPOCHS > 0, "Number of epochs must be positive"
    assert PPO_BATCH_SIZE > 0, "Batch size must be positive"
    assert PPO_BATCH_SIZE <= PPO_ROLLOUT_STEPS, \
        "Batch size must not exceed rollout steps"
    assert PPO_TOTAL_TIMESTEPS > PPO_ROLLOUT_STEPS, \
        "Total timesteps must exceed rollout steps"
    
    assert PREDICTION_LOSS_WEIGHT >= 0, "Prediction loss weight must be non-negative"
    assert PPO_LOSS_WEIGHT >= 0, "PPO loss weight must be non-negative"
    
    # ============================================================
    # Reward Function
    # ============================================================
    assert REWARD_PREDICTION_WEIGHT >= 0, \
        "Prediction reward weight must be non-negative"
    assert REWARD_TRACKING_WEIGHT >= 0, \
        "Tracking reward weight must be non-negative"
    assert REWARD_ERROR_SCALE > 0, "Reward error scale must be positive"
    assert 0 < REWARD_VEL_PREDICTION_WEIGHT_FACTOR <= 1.0, \
        "Velocity prediction weight factor must be in (0, 1]"
    
    # ============================================================
    # Deployment
    # ============================================================
    assert MAX_INFERENCE_TIME < CONTROL_CYCLE_TIME, \
        f"Inference time ({MAX_INFERENCE_TIME*1000:.2f}ms) must be less than " \
        f"control cycle ({CONTROL_CYCLE_TIME*1000:.2f}ms)"
    
    # ============================================================
    # Logging
    # ============================================================
    assert LOG_FREQ > 0, "Log frequency must be positive"
    assert SAVE_FREQ > 0, "Save frequency must be positive"
    assert CHECKPOINT_DIR != "", "Checkpoint directory must be specified"
    
    # ============================================================
    # Early Stopping
    # ============================================================
    if ENABLE_EARLY_STOPPING:
        assert EARLY_STOPPING_PATIENCE > 0, "Early stopping patience must be positive"
        assert EARLY_STOPPING_MIN_DELTA >= 0, "Early stopping min delta must be non-negative"
        assert EARLY_STOPPING_CHECK_FREQ > 0, "Early stopping check frequency must be positive"
    
    # ============================================================
    # Print Validation Summary
    # ============================================================
    print("\n" + "="*70)
    print("Configuration Validation Summary")
    print("="*70)
    
    print("\n✓ Robot Configuration:")
    print(f"  • Joints: {N_JOINTS}")
    print(f"  • Joint Limits: [{JOINT_LIMITS_LOWER[0]:.2f}, {JOINT_LIMITS_UPPER[0]:.2f}] rad (first joint)")
    print(f"  • Max Torque: {TORQUE_LIMITS[0]:.1f} Nm (first joint)")
    print(f"  • Max RL Compensation: {MAX_TORQUE_COMPENSATION[0]:.1f} Nm (first joint)")
    
    print("\n✓ Control System:")
    print(f"  • Frequency: {DEFAULT_CONTROL_FREQ} Hz")
    print(f"  • Control Cycle: {CONTROL_CYCLE_TIME*1000:.2f} ms")
    print(f"  • Max Inference Time: {MAX_INFERENCE_TIME*1000:.2f} ms")
    print(f"  • Safety Margin: {(CONTROL_CYCLE_TIME - MAX_INFERENCE_TIME)*1000:.2f} ms")
    
    print("\n✓ Environment:")
    print(f"  • Max Episode Steps: {MAX_EPISODE_STEPS}")
    print(f"  • Observation Dimension: {OBS_DIM}")
    print(f"  • Action Dimension: {N_JOINTS}")
    print(f"  • Target History Length: {TARGET_HISTORY_LEN}")
    print(f"  • Action History Length: {ACTION_HISTORY_LEN}")
    
    print("\n✓ Model Architecture:")
    print(f"  • LSTM Hidden Dim: {RNN_HIDDEN_DIM}")
    print(f"  • LSTM Layers: {RNN_NUM_LAYERS}")
    print(f"  • LSTM Sequence Length: {RNN_SEQUENCE_LENGTH}")
    print(f"  • Policy MLP: {PPO_MLP_HIDDEN_DIMS}")
    
    print("\n✓ Training Configuration:")
    print(f"  • Algorithm: Recurrent-PPO (End-to-End)")
    print(f"  • Total Timesteps: {PPO_TOTAL_TIMESTEPS:,}")
    print(f"  • Rollout Steps: {PPO_ROLLOUT_STEPS}")
    print(f"  • Epochs per Update: {PPO_NUM_EPOCHS}")
    print(f"  • Batch Size: {PPO_BATCH_SIZE}")
    print(f"  • Learning Rate: {PPO_LEARNING_RATE}")
    print(f"  • Total Updates: {PPO_TOTAL_TIMESTEPS // PPO_ROLLOUT_STEPS:,}")
    
    print("\n✓ Reward Configuration:")
    print(f"  • Prediction Weight: {REWARD_PREDICTION_WEIGHT}")
    print(f"  • Tracking Weight: {REWARD_TRACKING_WEIGHT}")
    print(f"  • Error Scale: {REWARD_ERROR_SCALE}")
    print(f"  • Velocity Factor: {REWARD_VEL_PREDICTION_WEIGHT_FACTOR}")
    
    print("\n✓ Loss Weights:")
    print(f"  • Prediction Loss: {PREDICTION_LOSS_WEIGHT}")
    print(f"  • PPO Loss: {PPO_LOSS_WEIGHT}")
    
    if ENABLE_EARLY_STOPPING:
        print("\n✓ Early Stopping:")
        print(f"  • Enabled: Yes")
        print(f"  • Patience: {EARLY_STOPPING_PATIENCE} checks")
        print(f"  • Min Delta: {EARLY_STOPPING_MIN_DELTA}")
        print(f"  • Check Frequency: every {EARLY_STOPPING_CHECK_FREQ} updates")
    else:
        print("\n✗ Early Stopping: Disabled")
    
    print("\n" + "="*70)
    print("All configuration parameters validated successfully! ✓")
    print("="*70 + "\n")


# Run validation when config is imported
_validate_config()