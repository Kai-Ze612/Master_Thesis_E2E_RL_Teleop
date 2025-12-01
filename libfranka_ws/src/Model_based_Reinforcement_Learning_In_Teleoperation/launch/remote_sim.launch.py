"""
Pure Simulation Deployment Launch.
Runs: Leader (local_node) -> Agent (agent_node) -> SimRemoteRobot (remote_robot_sim)
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig

def generate_launch_description():
    pkg_name = 'Model_based_Reinforcement_Learning_In_Teleoperation'
    
    # Arguments
    config = LaunchConfiguration('config', default=str(ExperimentConfig.LOW_DELAY.value))
    seed = LaunchConfiguration('seed', default='50')
    
    # Leader specific args
    trajectory_type = LaunchConfiguration('trajectory_type', default='figure_8')
    randomize_params = LaunchConfiguration('randomize_params', default='false')

    return LaunchDescription([
        # 1. Leader Robot (Trajectory Generator)
        # Your setup.py defines this as 'local_node'
        Node(
            package=pkg_name,
            executable='local_node', 
            name='leader_robot_publisher',
            output='screen',
            parameters=[{
                'trajectory_type': trajectory_type,
                'randomize_params': randomize_params
            }]
        ),

        # 2. Agent (The Brain)
        Node(
            package=pkg_name,
            executable='agent_node',
            name='agent_node',
            output='screen',
            parameters=[{'experiment_config': config, 'seed': seed}],
            remappings=[
                ('local_robot/joint_states', '/local_robot/joint_states'),
                # Agent listens to the 'fake' robot state published by the simulator
                ('remote_robot/joint_states', '/franka/joint_states') 
            ]
        ),

        # 3. Simulated Remote Robot (The Physics)
        # Your setup.py defines this as 'remote_robot_sim'
        Node(
            package=pkg_name,
            executable='remote_sim', 
            name='remote_sim',
            output='screen',
            parameters=[{'experiment_config': config, 'seed': seed}],
            remappings=[
                # Simulator publishes to this topic to mimic the real hardware driver
                ('/franka/joint_states', '/franka/joint_states')
            ]
        )
    ])