#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "mujoco_ros_msgs::mujoco_ros_msgs__rosidl_generator_py" for configuration ""
set_property(TARGET mujoco_ros_msgs::mujoco_ros_msgs__rosidl_generator_py APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(mujoco_ros_msgs::mujoco_ros_msgs__rosidl_generator_py PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libmujoco_ros_msgs__rosidl_generator_py.so"
  IMPORTED_SONAME_NOCONFIG "libmujoco_ros_msgs__rosidl_generator_py.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS mujoco_ros_msgs::mujoco_ros_msgs__rosidl_generator_py )
list(APPEND _IMPORT_CHECK_FILES_FOR_mujoco_ros_msgs::mujoco_ros_msgs__rosidl_generator_py "${_IMPORT_PREFIX}/lib/libmujoco_ros_msgs__rosidl_generator_py.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
