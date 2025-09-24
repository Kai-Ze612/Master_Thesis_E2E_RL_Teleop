## entry_points={'console_scripts': [...]}

This is the most critical part for making your nodes runnable. It creates a link between a simple command-line name and your Python script. 

The format for each entry is:

```bash
'executable_name = package_name.path.to.python_file:main_function'
```

executable_name: The short name you'll use with ros2 run (e.g., simple_move_node).

package_name.path.to.python_file: The full Python import path to your script.

:main_function: Tells ROS2 to run the main function inside that script.