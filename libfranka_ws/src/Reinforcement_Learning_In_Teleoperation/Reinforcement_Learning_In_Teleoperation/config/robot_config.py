"""
Shared Robot configuration setting
"""

import numpy as np

# Model paths
DEFAULT_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/multipanda_ros2/franka_description/mujoco/franka/scene.xml"

# RL model paths
DEFAULT_RL_MODEL_PATH = "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/Reinforcement_Learning_In_Teleoperation/Reinforcement_Learning_In_Teleoperation/rl_agent/rl_training_output"

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
DEFAULT_CONTROL_FREQ = 500

# Default publish frequency for robot state (Hz)
DEFAULT_PUBLISH_FREQ = 100

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
ACTION_HISTORY_LEN = 10
TARGET_HISTORY_LEN = 10
OBS_HISTORY_LEN = 10

MAX_TORQUE_COMPENSATION = np.array([
    10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0
], dtype=np.float32)

_OBS_PACKET_FEATURE_SIZE = N_JOINTS + N_JOINTS + 1

OBS_DIM = (
    # Current (delayed) state from the *most recent* packet
    N_JOINTS +              # delayed_remote_q
    N_JOINTS +              # delayed_remote_qd
    1 +                     # obs_delay_magnitude
    
    # Real-time target/goal
    N_JOINTS +              # realtime_target_q
    (N_JOINTS * TARGET_HISTORY_LEN) + # target_q_history
    (N_JOINTS * TARGET_HISTORY_LEN) + # target_qd_history
    
    # Action info
    1 +                     # act_delay_magnitude
    (N_JOINTS * ACTION_HISTORY_LEN) + # action_history
    
    # Observation History
    (_OBS_PACKET_FEATURE_SIZE * OBS_HISTORY_LEN) # obs_history
)

######################################

# Robot configuration validation
def _validate_config():
    """Validate configuration consistency on import."""
    assert JOINT_LIMITS_LOWER.shape == (N_JOINTS,), f"JOINT_LIMITS_LOWER must have shape ({N_JOINTS},)"
    assert JOINT_LIMITS_UPPER.shape == (N_JOINTS,), f"JOINT_LIMITS_UPPER must have shape ({N_JOINTS},)"
    assert TORQUE_LIMITS.shape == (N_JOINTS,), f"TORQUE_LIMITS must have shape ({N_JOINTS},)"
    assert INITIAL_JOINT_CONFIG.shape == (N_JOINTS,), f"INITIAL_JOINT_CONFIG must have shape ({N_JOINTS},)"
    assert KP_LOCAL_DEFAULT.shape == (N_JOINTS,), f"KP_LOCAL_DEFAULT must have shape ({N_JOINTS},)"
    assert KD_LOCAL_DEFAULT.shape == (N_JOINTS,), f"KD_LOCAL_DEFAULT must have shape ({N_JOINTS},)"
    assert KP_REMOTE_NOMINAL.shape == (N_JOINTS,), f"KP_REMOTE_NOMINAL must have shape ({N_JOINTS},)"
    assert KD_REMOTE_NOMINAL.shape == (N_JOINTS,), f"KD_REMOTE_NOMINAL must have shape ({N_JOINTS},)"
    assert 0 < JOINT_LIMIT_MARGIN < 0.5, "JOINT_LIMIT_MARGIN should be reasonable (0 < margin < 0.5 rad)"
    assert np.all(JOINT_LIMITS_LOWER < JOINT_LIMITS_UPPER), "All lower joint limits must be strictly less than upper joint limits."
    assert np.all(INITIAL_JOINT_CONFIG >= JOINT_LIMITS_LOWER) and \
       np.all(INITIAL_JOINT_CONFIG <= JOINT_LIMITS_UPPER), \
       "Initial joint configuration must be within the joint limits."
    assert np.all(TORQUE_LIMITS >= 0), "Torque limits must be non-negative."
    assert np.all(KP_LOCAL_DEFAULT >= 0), "Local Kp gains must be non-negative."
    assert np.all(KD_LOCAL_DEFAULT >= 0), "Local Kd gains must be non-negative."
    assert TCP_OFFSET.shape == (3,), "TCP_OFFSET must be a 3D vector."
    assert TRAJECTORY_CENTER_DEFAULT.shape == (3,), "TRAJECTORY_CENTER_DEFAULT must be a 3D vector."
    print("Robot configuration validated successfully")


# Run validation when config is imported
_validate_config()