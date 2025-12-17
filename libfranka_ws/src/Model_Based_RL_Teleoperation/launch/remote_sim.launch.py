"""
Pure Simulation Deployment Launch.
Runs: Leader (local_node) -> Agent (agent_node) -> SimRemoteRobot (remote_robot_sim)
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from Model_based_RL_Teleoperation.utils.delay_simulator import ExperimentConfig

def generate_launch_description():
    pkg_name = 'Model_based_Reinforcement_Learning_In_Teleoperation'
    
    # Arguments
    config = LaunchConfiguration('config', default=str(ExperimentConfig.LOW_DELAY.value))
    seed = LaunchConfiguration('seed', default='50')
    
    return LaunchDescription([


        # 3. Simulated Remote Robot (The Physics)
        Node(
            package=pkg_name,
            executable='remote_sim', 
            name='remote_sim',
            output='screen',
            parameters=[{'experiment_config': config, 'seed': seed}],
            # REMOVED REMAPPINGS
        )
    ])