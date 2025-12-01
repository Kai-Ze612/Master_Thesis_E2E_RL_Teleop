# This script is to create packages for different experiments
# The new packages will be located under src directory of the workspace

#!/bin/bash
set -e  # Exit on any error

echo "Creating new packages"

# Define paths
MASTER_THESIS_PATH="/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws"
FRANKA_WS_PATH="$MASTER_THESIS_PATH/src"

# # Create workspace structure
# This line is only used for creating a new workspace
# echo "Creating workspace structure..."
# mkdir -p $FRANKA_WS_PATH/{src,build,install,log}
# cd $FRANKA_WS_PATH/src

source /opt/ros/humble/setup.bash

# Create package
## Modify here if want to use CPP - change ament_python to ament_cmake
echo "Creating ROS2 package..."
ros2 pkg create --build-type ament_python End-to-End_Teleoperation \
  --dependencies rclpy sensor_msgs std_msgs geometry_msgs

cd End-to-End_Teleoperation

mkdir -p End-to-End_Teleoperation/{nodes,controllers,utils}
mkdir -p {launch,config,models,meshes,textures,urdf,test,resource}

touch End-to-End_Teleoperation/resource/End-to-End_Teleoperation
touch End-to-End_Teleoperation/__init__.py
touch End-to-End_Teleoperation/nodes/__init__.py
touch End-to-End_Teleoperation/controllers/__init__.py
touch End-to-End_Teleoperation/utils/__init__.py