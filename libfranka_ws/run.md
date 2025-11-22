## Building Packages

1. Build the packages
```bash
cd /media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws
```

```bash
rm -rf build install log
```

```bash
colcon build --packages-up-to mujoco_ros
```

```bash
colcon build --symlink-install --packages-skip mujoco_ros mujoco_ros_msgs
```

To rebuild:
```bash
colcon build --symlink-install --packages-select 
```

symlink-install method is only for python script, the built files will connect with the original python files automatically, after modifying python file, we don't need to rebuild everytime.

If we build with CPP, we have to rebuild after evertime change, this is the difference.

```bash
source ./install/setup.bash
```

## Initial Start
Run the following commands after each start

```bash
cd /media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws
```

```bash
source setup_environment.sh
```

### Initial Position Return
```bash
cd ~/Libraries/libfranka/build/examples
./communication_test 192.168.03.107
```

----

## Run the real robot arm
1. Start robot with necessary features via bringup, this will load the controllers.
```bash
ros2 launch Model_based_Reinforcement_Learning_In_Teleoperation remote_agent.launch.py robot_ip:=192.168.03.108
```

2. Start local robot
```bash
ros2 launch Model_based_Reinforcement_Learning_In_Teleoperation local_robot.launch.py seed:=50 config:=1
```


---
## Run the simulation robot arm
1. Launch Simulator
```bash 
ros2 launch Reinforcement_Learning_In_Teleoperation teleoperation.launch.py
```

2. Activate Controller
```bash
ros2 control load_controller multi_mode_controller --set-state active
ros2 topic pub --once /local_robot/start_command std_msgs/msg/Bool '{data: true}'
```

3. Test: Publish Positional Command
``` bash
ros2 topic pub --once local_robot/cartesian_commands geometry_msgs/msg/Point '{x: -0.1, y: -0.1, z: 0.6}'
```
```bash
ros2 topic echo --once /local_robot/leader_pose
```


---
## Docs introduction:
1. `franka_bringup` is to start the robot with necessary features, it serves like a power up for a car (start engine, bashboard....)
```bash
ros2 launch franka_bringup franka.launch.py robot_ip:=192.168.03.107
```
2. `franka_description` is the blueprint of the robot, it tells simulator and ROS2 system/Motion Planning the structure of the robot. We don't use it directly, ros2 launch will start it automatically.
3. `franka_hardware` is the driver. It's the low-level bridge that connects ros2_control to the hardware. We don't use it directly. ros2 launch will start it automatically.
4. `multi_mode_controller`  This is the high-level controller. It acts as a manager that can load and switch between simpler, specialized controllers (called "controllets"), such as a joint impedance controller or the Cartesian impedance controller you've been using. We have to use this file directly in our python.