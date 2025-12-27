"""
Launch file for VIRTUAL EXPERIMENT (Simulation + MuJoCo Viewer).
Launches:
1. Local Robot (Trajectory Generator)
2. Remote Robot (MuJoCo Simulator + Viewer)
3. Agent Node (RL Control Policy)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    my_package_name = 'E2E_Teleoperation'

    # --- Arguments ---
    # Experiment Config
    config = LaunchConfiguration('config')
    seed = LaunchConfiguration('seed')
    
    # Trajectory Config
    trajectory_type = LaunchConfiguration('trajectory_type')
    randomize_trajectory = LaunchConfiguration('randomize_trajectory')

    ld = LaunchDescription()

    # --- Declarations ---
    ld.add_action(DeclareLaunchArgument(
        'config', 
        default_value='3', 
        description='Experiment config (1=LOW, 2=HIGH, 3=FULL/VAR)'
    ))
    ld.add_action(DeclareLaunchArgument(
        'seed', 
        default_value='50', 
        description='Random seed'
    ))
    ld.add_action(DeclareLaunchArgument(
        'trajectory_type', 
        default_value='figure_8', 
        description='Trajectory type (figure_8, square, lissajous_complex)'
    ))
    ld.add_action(DeclareLaunchArgument(
        'randomize_trajectory', 
        default_value='false', 
        description='Randomize trajectory parameters'
    ))

    # --- 1. Leader Robot (Trajectory Generator) ---
    ld.add_action(Node(
        package=my_package_name,
        executable='local_robot',  # Ensure this matches setup.py entry_point
        name='leader_robot_node',
        output='screen',
        parameters=[{
            'trajectory_type': trajectory_type,
            'randomize_params': randomize_trajectory
        }]
    ))

    # --- 2. Remote Robot (Simulator + Viewer) ---
    ld.add_action(Node(
        package=my_package_name,
        executable='remote_node',  # Ensure this matches setup.py entry_point
        name='remote_robot_node',
        output='screen',
        parameters=[{
            'experiment_config': config,
            'seed': seed
        }]
    ))

    # --- 3. Agent Node (RL Brain) ---
    ld.add_action(Node(
        package=my_package_name,
        executable='agent_node',   # Ensure this matches setup.py entry_point
        name='agent_node',
        output='screen',
        parameters=[{
            'experiment_config': config, 
            'seed': seed                  
        }],
        # Explicit Remappings
        remappings=[
            ('local_robot/joint_states', '/local_robot/joint_states'),
            ('remote_robot/joint_states', '/remote_robot/joint_states'), 
            ('agent/tau_rl', '/agent/tau_rl'),
            ('agent/predict_target', '/agent/predict_target')
        ]
    ))

    return ld