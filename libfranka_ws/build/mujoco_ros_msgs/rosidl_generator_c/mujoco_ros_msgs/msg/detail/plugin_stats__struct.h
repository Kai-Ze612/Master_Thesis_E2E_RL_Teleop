// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/PluginStats.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'plugin_type'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/PluginStats in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__msg__PluginStats
{
  rosidl_runtime_c__String plugin_type;
  float load_time;
  float reset_time;
  float ema_steptime_control;
  float ema_steptime_passive;
  float ema_steptime_render;
  float ema_steptime_last_stage;
} mujoco_ros_msgs__msg__PluginStats;

// Struct for a sequence of mujoco_ros_msgs__msg__PluginStats.
typedef struct mujoco_ros_msgs__msg__PluginStats__Sequence
{
  mujoco_ros_msgs__msg__PluginStats * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__PluginStats__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__STRUCT_H_
