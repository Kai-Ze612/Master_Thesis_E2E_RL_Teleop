from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    my_package_name = 'Reinforcement_Learning_In_Teleoperation'

    # --- Declare Trajectory Arguments ---
    trajectory_type_param_name = 'trajectory_type'
    randomize_trajectory_param_name = 'randomize_trajectory'

    trajectory_type = LaunchConfiguration(trajectory_type_param_name)
    randomize_trajectory = LaunchConfiguration(randomize_trajectory_param_name)

    ld = LaunchDescription()

    # --- Add Arguments ---
    ld.add_action(DeclareLaunchArgument(
        trajectory_type_param_name,
        default_value='figure_8',
        description='Type of trajectory for the leader (figure_8, square, etc.)'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        randomize_trajectory_param_name,
        default_value='false',
        description='Whether to randomize the trajectory parameters'
    ))

    ld.add_action(Node(
        package=my_package_name,
        executable='local_node',
        name='leader_robot_publisher',
        output='screen',
        parameters=[{
            'trajectory_type': trajectory_type,
            'randomize_params': randomize_trajectory
        }]
    ))
    
    return ld