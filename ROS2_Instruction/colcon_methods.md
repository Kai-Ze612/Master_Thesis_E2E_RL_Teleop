## 1. colcon build (The Full Rebuild)
This is the standard command to build your entire ROS2 workspace.

What it does: It goes through every single package in your src directory, compiles all the C++ code, processes all the Python files, and copies all the necessary files (launch files, YAML configs, etc.) into the install directory.

Note:
* Run colcon build only at the first time
* Run only after a major package change

## 2. colcon build --packages-select <package_name> (The Targeted Fix)
This command builds only the specific package(s) you name.

What it does: It tells colcon to ignore every other package and only process the one(s) you list. This is extremely fast if you've only made changes to one or two packages.

Note:
* After first time built, run this with only changed package
* When using C++, always have to use this
* Wehn change setup.py or other system level files, have to run this command

## 3. colcon build --symlink-install (The Smart Shortcut)
This works only for python

What it does: For Python files, instead of copying them from src to install, it creates a symbolic link (a shortcut). This means the file in the install directory just points directly to your original file in src.

Note:
* Once we have run this one time, we can change python file without colcon again
* Only run when we add more files
