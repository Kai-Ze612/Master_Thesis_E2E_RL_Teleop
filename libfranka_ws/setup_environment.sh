
# The setup environment script for Master thesis
# Should run every time you want to work on the project.

#!/bin/bash


# Master Thesis ROS2 Environment Setup
export MASTER_THESIS_PATH="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation"
export FRANKA_WS_PATH="$MASTER_THESIS_PATH/libfranka_ws"

# Make sure conda is deactivated
conda deactivate 

# Source ROS2
source /opt/ros/humble/setup.bash

# Source workspace if built
source "$FRANKA_WS_PATH/install/setup.bash"

# MuJoCo environment
export MUJOCO_PATH=/home/kai/Libraries/mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kai/Libraries/mujoco/lib

# Change to workspace directory
cd "$FRANKA_WS_PATH"
echo "Environment Ready!"
echo "Workspace: $FRANKA_WS_PATH"