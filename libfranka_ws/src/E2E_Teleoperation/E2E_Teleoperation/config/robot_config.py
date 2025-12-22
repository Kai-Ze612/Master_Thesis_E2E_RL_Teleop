
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

######################################
# 1. GLOBAL CONSTANTS
######################################

# --- File Paths ---
CONFIG_FILE_PATH = Path(__file__).resolve()
CONFIG_DIR = CONFIG_FILE_PATH.parent
PACKAGE_ROOT = CONFIG_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_SRC = PROJECT_ROOT.parent

CHECKPOINT_DIR = PACKAGE_ROOT / "trained_RL"
LOG_DIR = PACKAGE_ROOT / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MUJOCO_MODEL_PATH = (
    WORKSPACE_SRC / "multipanda_ros2" / "franka_description" / "mujoco" / "franka" / "scene.xml"
)
DEFAULT_RL_MODEL_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/E2E_Teleoperation/E2E_Teleoperation/trained_RL/E2E_RL_HIGH_VARIANCE_FIGURE_8_20251221_224820/best_policy_stage3.pth"

# --- Franka Panda Parameters ---
N_JOINTS = 7
EE_BODY_NAME = "panda_hand"
TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float32)

JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.5445, -3.0159], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.5169, 3.0159], dtype=np.float32)
JOINT_LIMIT_MARGIN = 0.05

TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float32)
MAX_ACTION_TORQUE = TORQUE_LIMITS.copy()

INITIAL_JOINT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.5708, 0.785], dtype=np.float32)

# --- Normalization ---
Q_MEAN = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78], dtype=np.float32)
Q_STD  = np.array([1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32) 
QD_MEAN = np.zeros(7, dtype=np.float32)
QD_STD  = np.ones(7, dtype=np.float32) * 2.0
DELAY_INPUT_NORM_FACTOR = 100.0

# --- Simulation & Control ---
CONTROL_FREQ = 250
DT = 1.0 / CONTROL_FREQ
WARM_UP_DURATION = 0.1
NO_DELAY_DURATION = 0.1
MAX_EPISODE_STEPS = 5500
MAX_JOINT_ERROR_TERMINATION = 2.0

# --- IK Solver Parameters ---
IK_POSITION_TOLERANCE = 0.01
IK_JACOBIAN_MAX_ITER = 300
IK_OPTIMIZATION_MAX_ITER = 100
IK_JACOBIAN_STEP_SIZE = 0.01
IK_JACOBIAN_DAMPING = 0.1
IK_NULL_SPACE_GAIN = 0.5

# --- Trajectory Generation ---
TRAJECTORY_CENTER = np.array([0.3, 0, 0.5], dtype=np.float32)
TRAJECTORY_SCALE = np.array([0.2, 0.2, 0.02], dtype=np.float32)
TRAJECTORY_FREQUENCY = 0.1

# --- Network Architecture ---
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 3
RNN_SEQUENCE_LENGTH = 80
ESTIMATOR_STATE_DIM = 15
ESTIMATOR_OUTPUT_DIM = 14
MLP_HIDDEN_DIMS = [512, 256]
LOG_STD_MIN = -20
LOG_STD_MAX = 2

ROBOT_STATE_DIM = 14
ROBOT_HISTORY_DIM = RNN_SEQUENCE_LENGTH * ROBOT_STATE_DIM
TARGET_HISTORY_DIM = RNN_SEQUENCE_LENGTH * ESTIMATOR_STATE_DIM
OBS_DIM = ROBOT_STATE_DIM + ROBOT_HISTORY_DIM + TARGET_HISTORY_DIM

# --- Training Hyperparameters ---
SEED = 42
BATCH_SIZE = 1024
BUFFER_SIZE = 1_000_000
STAGE1_STEPS = 20_000
ENCODER_LR = 1e-4
STAGE2_TOTAL_STEPS = 200_000
STAGE3_TOTAL_STEPS = 1_000_000
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
LOG_FREQ = 50
VAL_FREQ = 5000

# --- Teacher Gains (TUNED FOR DELAY STABILITY) ---
TEACHER_KP = np.array([144.0, 144.0, 144.0, 144.0, 144.0, 144.0, 20.0], dtype=np.float64)
TEACHER_KD = np.array([24.0,  24.0,  24.0,  24.0,  24.0,  24.0,  4.0], dtype=np.float64)
TEACHER_SMOOTHING = 0.5

######################################
# 2. DATACLASSES
######################################

@dataclass(frozen=True)
class RobotConfig:
    N_JOINTS: int = N_JOINTS
    CONTROL_FREQ: int = CONTROL_FREQ
    DT: float = DT
    TORQUE_LIMITS: np.ndarray = field(default_factory=lambda: TORQUE_LIMITS)
    MAX_ACTION_TORQUE: np.ndarray = field(default_factory=lambda: MAX_ACTION_TORQUE)
    JOINT_LIMITS_LOWER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_LOWER)
    JOINT_LIMITS_UPPER: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_UPPER)
    Q_MEAN: np.ndarray = field(default_factory=lambda: Q_MEAN)
    Q_STD: np.ndarray = field(default_factory=lambda: Q_STD)
    QD_MEAN: np.ndarray = field(default_factory=lambda: QD_MEAN)
    QD_STD: np.ndarray = field(default_factory=lambda: QD_STD)
    TRAJECTORY_CENTER: np.ndarray = field(default_factory=lambda: TRAJECTORY_CENTER)
    TRAJECTORY_SCALE: np.ndarray = field(default_factory=lambda: TRAJECTORY_SCALE)
    TRAJECTORY_FREQUENCY: float = TRAJECTORY_FREQUENCY
    RNN_SEQ_LEN: int = RNN_SEQUENCE_LENGTH
    RNN_HIDDEN_DIM: int = RNN_HIDDEN_DIM
    RNN_NUM_LAYERS: int = RNN_NUM_LAYERS
    ROBOT_STATE_DIM: int = ROBOT_STATE_DIM
    ESTIMATOR_INPUT_DIM: int = ESTIMATOR_STATE_DIM
    ESTIMATOR_OUTPUT_DIM: int = ESTIMATOR_OUTPUT_DIM
    ESTIMATOR_STATE_DIM: int = ESTIMATOR_STATE_DIM
    ROBOT_HISTORY_DIM: int = ROBOT_HISTORY_DIM
    TARGET_HISTORY_DIM: int = TARGET_HISTORY_DIM
    OBS_DIM: int = OBS_DIM
    PROJECT_ROOT: Path = PROJECT_ROOT
    CHECKPOINT_DIR: Path = CHECKPOINT_DIR
    LOG_DIR: Path = LOG_DIR
    DEFAULT_MUJOCO_MODEL_PATH: Path = DEFAULT_MUJOCO_MODEL_PATH
    MAX_EPISODE_STEPS: int = MAX_EPISODE_STEPS
    MAX_JOINT_ERROR_TERMINATION: float = MAX_JOINT_ERROR_TERMINATION
    INITIAL_JOINT_CONFIG: np.ndarray = field(default_factory=lambda: INITIAL_JOINT_CONFIG)
    WARM_UP_DURATION: float = WARM_UP_DURATION
    NO_DELAY_DURATION: float = NO_DELAY_DURATION


@dataclass
class TrainConfig:
    SEED: int = SEED
    BATCH_SIZE: int = BATCH_SIZE
    BUFFER_SIZE: int = BUFFER_SIZE
    GAMMA: float = GAMMA
    STAGE1_STEPS: int = STAGE1_STEPS
    STAGE2_STEPS: int = STAGE2_TOTAL_STEPS
    STAGE3_STEPS: int = STAGE3_TOTAL_STEPS
    ENCODER_LR: float = ENCODER_LR
    ACTOR_LR: float = ACTOR_LR
    CRITIC_LR: float = CRITIC_LR
    ALPHA_LR: float = ALPHA_LR
    LOG_FREQ: int = LOG_FREQ
    VAL_FREQ: int = VAL_FREQ


@dataclass
class SACConfig:
    """
    Stage 3 SAC Fine-tuning Hyperparameters
    
    UPDATED: Stronger BC to prevent policy collapse
    """
    
    # Warmup
    WARMUP_STEPS: int = 10000
    
    # Reward (NO scaling to prevent Q explosion)
    REWARD_SCALE: float = 1.0
    
    # Behavioral Cloning Regularization
    # KEY CHANGES: Much slower decay, higher minimum
    BC_DECAY_STEPS: int = 200000      # Was 50000 - now 4x slower
    BC_INITIAL_WEIGHT: float = 5.0    # Same
    BC_MIN_WEIGHT: float = 2.0        # Was 0.1 - now keeps strong BC throughout
    
    # Target Network (slower update for stability)
    TARGET_TAU: float = 0.001
    
    # TD3-style Delayed Policy Updates (disabled for simplicity)
    POLICY_DELAY: int = 1  # Update every step
    
    # Q-value Clipping (prevents explosion)
    Q_CLIP_MAX: float = 100.0
    
    # Gradient Clipping (tighter for stability)
    GRAD_CLIP_CRITIC: float = 1.0
    GRAD_CLIP_ACTOR: float = 1.0
    
    # Entropy Tuning
    TARGET_ENTROPY_RATIO: float = 0.5
    ALPHA_LR: float = 1e-4


######################################
# 3. INSTANTIATE CONFIGS
######################################

ROBOT = RobotConfig()
TRAIN = TrainConfig()
SAC = SACConfig()