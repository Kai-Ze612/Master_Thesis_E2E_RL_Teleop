# garmi_hardware

This package implements a UDP-based `ros2-control` hardware interface, for communicating with Garmi's head and base virtual machines. 
It should be executed together with the respective hardware's main software in those virtual machines.

Environment variables are important for making sure the package works. They are:

- `HEAD_VM_IP`: IP address of the head's virtual machine.
- `BASE_VM_IP`: IP address of the base's virtual machine.
- `HEAD_VM_SEND_PORT`: UDP port to send command packets to, on the head virtual machine. It should match the port of the UDP receiver on the VM.
- `BASE_VM_SEND_PORT`: Ditto, for base.
- `HEAD_VM_RECV_PORT`: UDP port to receive state packets from, running on the local machine. It should match the port of the UDP sender on the VM. 
- `BASE_VM_RECV_PORT`: Ditto, for base.

`garmi_base_hardware_interface` requires all the`BASE_` vars to be set, and likewise for `garmi_head_hardware_interface`.
If you want to test the interface code without having to start up a UDP sender, make sure to comment out this code line in both `base` and `head`'s `.cpp` files:
```
udp_server_->recv_udp_packet();
```
Since this function will cause the code to hang until it receives something.

## enums
There are two enums that I use for managing both the head and the base data, since they both have two motors: `FrankaMotorState` and `FrankaMotorCommand`. 
They both have the name convention of `l_` and `r_`, for left and right. For base, that makes sense, but not really for head.
In the head case, just think in alphabetical order, I guess... `l` comes before `r`, so `l` is pan (index 0), and `r` is tilt (index 1).

## garmi_description
I added new robot description, one for the whole system (arm, base, head) and another for just the `garmi_hardware` parts (base, head).
The whole system ones are:
- `garmi_base_real.urdf.xacro`, which calls
    - the dual arm description,
    - `garmi_base_real.ros2_control.xacro`
    - `garmi_head_real.ros2_control.xacro`
- `garmi_base_head_only_real.urdf.xacro`, which calls
    - `garmi_base_real.ros2_control.xacro`
    - `garmi_head_real.ros2_control.xacro`

The latter is mainly for testing. These two `ros2_control.xacro` files then start up the `garmi_hardware`'s interfaces, so you can test just the head and the base separately.

## garmi_bringup
New launch files have been added too. They are:
- `launch/real/garmi.launch.py`
    - This then calls the full Garmi xacro, i.e. `garmi_base_real.urdf.xacro`
- `launch/real/base_head_only.launch.py`
    - This, as the name suggests, calls the base and head-only xacro, `garmi_base_head_only_real.urdf.xacro`

## Remaining TODOs
- Ensure that the integer-converted values of `enum class franka_joint_driver::Error` and `enum garmi_hardware::HardwareID` are identical on both Garmi2 and the virtual machines
- Check whether executing `recv_udp_packet()` directly in the `read(...)` functions of `garmi_(base/head)_hardware_interface` doesn't lead to any issues, and if so, see if adding a thread can fix the issue
- Implement the VM-side main software including the UDP stuff, and make sure that the VM-side environment variables are integrated properly. They are:
    - For the head vm,
        - `GARMI2_IP`: IP of the garmi2
        - `GARMI2_HEAD_SEND_PORT` Port to send the state messages to. This should match `HEAD_VM_RECV_PORT`.
        - `GARMI2_HEAD_RECV_PORT` Port to receive the command messages from. This should match `HEAD_VM_SEND_PORT`.
    - For the base vm,
        - `GARMI2_IP`: IP of the garmi2
        - `GARMI2_BASE_SEND_PORT` Port to send the state messages to. This should match `BASE_VM_RECV_PORT`.
        - `GARMI2_BASE_RECV_PORT` Port to receive the command messages from. This should match `BASE_VM_SEND_PORT`.
- Check whether `joint_state` is being mapped properly, so that the rviz visualization works as intended.
    - The current setup in `base_head_only.launch.py` changes the head positions correctly in RVIZ (e.g. when writing `l_position_ = 1.57` to turn the head), but the joint states may need to be mapped more properly when the arms come into the picture.
    - This has mostly to do with the way the topics need to be mapped in these lines in the launch files:
    ```
    Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[
                {'source_list': ['garmi_base_head/joint_states'], <-- here
                 'rate': 30}],
        ),
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[{'robot_description': robot_description}, franka_controllers],
            remappings=[('joint_states', 'garmi_base_head/joint_states')], <-- here
            output={
                'stdout': 'screen',
                'stderr': 'screen',
            },
            on_exit=Shutdown(),
        ),
    ```
    - My guess: Since the entire `xacro` file is loaded at once and managed by `ros2_control_node`, all the joints, i.e. arms, head, base will all be broadcast to `joint_states`, which is then remapped to e.g. `franka/joint_states`.