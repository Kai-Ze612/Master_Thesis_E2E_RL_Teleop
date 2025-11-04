import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # --- 1. Declare Launch Arguments ---
    # This allows you to change parameters from the command line
    # e.g., ros2 launch ... experiment_config:='2'
    
    # Argument for the leader's trajectory
    declared_trajectory_type = DeclareLaunchArgument(
        'trajectory_type',
        default_value='figure_8',
        description='Type of trajectory for the leader (figure_8, square, etc.)'
    )

    # Argument for the agent's delay configuration
    declared_experiment_config = DeclareLaunchArgument(
        'experiment_config',
        default_value='3', # '3' for HIGH_DELAY
        description='Delay config for the agent (1=LOW, 2=MEDIUM, 3=HIGH)'
    )
    
    # Argument for trajectory randomization
    declared_randomize_trajectory = DeclareLaunchArgument(
        'randomize_trajectory',
        default_value='false',
        description='Whether to randomize the trajectory parameters'
    )

    # --- 2. Get Launch Configurations ---
    # These reference the arguments declared above
    trajectory_type = LaunchConfiguration('trajectory_type')
    experiment_config = LaunchConfiguration('experiment_config')
    randomize_trajectory = LaunchConfiguration('randomize_trajectory')

    # --- 3. Define Nodes ---

    # Node 1: Leader Robot (Trajectory Generator)
    local_node = Node(
        package='Reinforcement_Learning_In_Teleoperation',
        executable='local_node',
        name='leader_robot_publisher',
        output='screen',
        parameters=[{
            'trajectory_type': trajectory_type,
            'randomize_params': randomize_trajectory
        }]
    )

    # Node 2: Agent (The "Brain")
    agent_node = Node(
        package='Reinforcement_Learning_In_Teleoperation',
        executable='agent_node',
        name='agent_node',
        output='screen',
        parameters=[{
            'experiment_config': experiment_config
            # Note: The agent_path is hard-coded in your agent.py
            # If you wanted to, you could make that a launch parameter too.
        }]
    )

    # Node 3: Remote Robot (The "Body")
    remote_node = Node(
        package='Reinforcement_Learning_In_Teleoperation',
        executable='remote_node',
        name='remote_robot_node',
        output='screen'
        # This node doesn't need parameters as its config is
        # from robot_config.py
    )

    # --- 4. Create Launch Description ---
    return LaunchDescription([
        # Add the declared arguments
        declared_trajectory_type,
        declared_experiment_config,
        declared_randomize_trajectory,
        
        # Add the nodes
        local_node,
        agent_node,
        remote_node
    ])