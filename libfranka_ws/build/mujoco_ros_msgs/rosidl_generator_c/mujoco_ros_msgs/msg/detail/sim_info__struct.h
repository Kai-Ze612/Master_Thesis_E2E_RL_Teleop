// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/SimInfo.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'model_path'
#include "rosidl_runtime_c/string.h"
// Member 'loading_state'
#include "mujoco_ros_msgs/msg/detail/state_uint__struct.h"

/// Struct defined in msg/SimInfo in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__msg__SimInfo
{
  rosidl_runtime_c__String model_path;
  bool model_valid;
  /// counter of (re)loads
  uint16_t load_count;
  mujoco_ros_msgs__msg__StateUint loading_state;
  bool paused;
  uint16_t pending_sim_steps;
  /// measured real-time factor
  float rt_measured;
  /// desired real-time factor
  float rt_setting;
} mujoco_ros_msgs__msg__SimInfo;

// Struct for a sequence of mujoco_ros_msgs__msg__SimInfo.
typedef struct mujoco_ros_msgs__msg__SimInfo__Sequence
{
  mujoco_ros_msgs__msg__SimInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__SimInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__STRUCT_H_
