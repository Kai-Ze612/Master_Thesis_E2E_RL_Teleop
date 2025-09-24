#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "mujoco_ros2_control_system::mujoco_ros2_control_system" for configuration ""
set_property(TARGET mujoco_ros2_control_system::mujoco_ros2_control_system APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(mujoco_ros2_control_system::mujoco_ros2_control_system PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libmujoco_ros2_control_system.so"
  IMPORTED_SONAME_NOCONFIG "libmujoco_ros2_control_system.so"
  )

list(APPEND _cmake_import_check_targets mujoco_ros2_control_system::mujoco_ros2_control_system )
list(APPEND _cmake_import_check_files_for_mujoco_ros2_control_system::mujoco_ros2_control_system "${_IMPORT_PREFIX}/lib/libmujoco_ros2_control_system.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
