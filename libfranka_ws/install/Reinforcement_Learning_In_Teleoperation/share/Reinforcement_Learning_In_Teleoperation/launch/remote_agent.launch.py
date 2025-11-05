import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    my_package_name = 'Reinforcement_Learning_In_Teleoperation'

    # --- Franka Hardware Arguments ---
    robot_ip_parameter_name = 'robot_ip'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'
    use_rviz_parameter_name = 'use_rviz'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)

    # --- Setup Robot Description & Controllers (from Franka) ---
    franka_xacro_file = os.path.join(get_package_share_directory('franka_description'), 'robots', 'real',
                                     'panda_arm.urdf.xacro')
    robot_description = Command(
        [FindExecutable(name='xacro'), ' ', franka_xacro_file, ' hand:=', load_gripper,
         ' robot_ip:=', robot_ip, ' use_fake_hardware:=', use_fake_hardware,
         ' fake_sensor_commands:=', fake_sensor_commands])

    rviz_file = os.path.join(get_package_share_directory('franka_description'), 'rviz',
                             'visualize_franka.rviz')

    franka_controllers = PathJoinSubstitution(
        [
            FindPackageShare('franka_bringup'),
            'config',
            'real',
            'single_controllers.yaml', # This file contains 'joint_tau_controller'
        ]
    )

    # --- Start Launch Description ---
    ld = LaunchDescription()

    # --- Add All Launch Arguments ---
    ld.add_action(DeclareLaunchArgument(
        robot_ip_parameter_name,
        description='Hostname or IP address of the robot.'))
    ld.add_action(DeclareLaunchArgument(
        use_rviz_parameter_name,
        default_value='false',
        description='Visualize the robot in Rviz'))
    ld.add_action(DeclareLaunchArgument(
        use_fake_hardware_parameter_name,
        default_value='false',
        description='Use fake hardware'))
    ld.add_action(DeclareLaunchArgument(
        fake_sensor_commands_parameter_name,
        default_value='false',
        description="Fake sensor commands. Only valid when '{}' is true".format(
            use_fake_hardware_parameter_name)))
    ld.add_action(DeclareLaunchArgument(
        load_gripper_parameter_name,
        default_value='false',
        description='Use Franka Gripper as an end-effector.'))

    # --- Add Franka Hardware Nodes ---
    
    ld.add_action(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    ))
    
    ld.add_action(Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'source_list': ['franka/joint_states', 'panda_gripper/joint_states'],
             'rate': 30}],
    ))

    ld.add_action(Node(
        package='franka_control2',
        executable='franka_control2_node',
        parameters=[{'robot_description': robot_description}, franka_controllers],
        remappings=[('joint_states', 'franka/joint_states')],
        output='screen',
        on_exit=Shutdown(),
    ))

    # --- Spawn ALL necessary controllers ---
    ld.add_action(Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    ))

    ld.add_action(Node(
        package='controller_manager',
        executable='spawner',
        arguments=['franka_robot_state_broadcaster'],
        output='screen',
        condition=UnlessCondition(use_fake_hardware),
    ))

    ld.add_action(Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_tau_controller'], # Use the controller from your YAML
        output='screen',
    ))

    # ... (Gripper and RViz nodes) ...
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution(
            [FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'])]),
        launch_arguments={robot_ip_parameter_name: robot_ip,
                          use_fake_hardware_parameter_name: use_fake_hardware}.items(),
        condition=IfCondition(load_gripper)
    ))
    ld.add_action(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['--display-config', rviz_file],
        condition=IfCondition(use_rviz)
    ))

    # --- Add Your Debug Remote Node (Agent is commented out) ---

    # ld.add_action(Node(
    #     package=my_package_name,
    #     executable='agent_node',
    #     ...
    # ))

    # CRITICAL FIX: Removed incorrect torque command remapping
    # The controller listens to /joint_tau/torques_desired, not /joint_tau_controller/commands
    ld.add_action(Node(
        package=my_package_name,
        executable='remote_node',
        name='remote_robot_node',
        output='screen',
        remappings=[
            # Subscribe to the actual hardware joint states topic
            ('remote_robot/joint_states', '/franka/joint_states'),
            # Subscribe to the leader robot's joint states
            ('local_robot/joint_states', '/local_robot/joint_states'),
            # NO remapping for torques - publish directly to /joint_tau/torques_desired
        ]
    ))

    return ld