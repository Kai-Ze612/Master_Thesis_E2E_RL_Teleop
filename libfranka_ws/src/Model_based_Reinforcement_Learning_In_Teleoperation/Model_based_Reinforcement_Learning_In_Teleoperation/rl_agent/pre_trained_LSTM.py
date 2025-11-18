"""
Pre-training script for the State Estimator (LSTM).

pipelines:
1. Collect data by running the TeleoperationEnvWithDelay environment with a random policy.
2. Store (delayed_sequence, true_target) pairs in a replay buffer.
3. Train the StateEstimator LSTM in a supervised learning. (min MSE loss)
"""

import os
import sys
import torch
import numpy as np
import argparse
import logging
from datetime import datetime
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple

from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import (
    TrajectoryType,
    TrajectoryParams,
    Figure8TrajectoryGenerator,
    SquareTrajectoryGenerator,
    LissajousComplexGenerator,
    TrajectoryGenerator
)

