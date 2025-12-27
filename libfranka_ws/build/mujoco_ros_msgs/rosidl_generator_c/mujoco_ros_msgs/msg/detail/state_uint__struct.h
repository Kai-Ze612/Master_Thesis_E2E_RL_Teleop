// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/StateUint.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__STATE_UINT__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__STATE_UINT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'description'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/StateUint in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__msg__StateUint
{
  uint8_t value;
  rosidl_runtime_c__String description;
} mujoco_ros_msgs__msg__StateUint;

// Struct for a sequence of mujoco_ros_msgs__msg__StateUint.
typedef struct mujoco_ros_msgs__msg__StateUint__Sequence
{
  mujoco_ros_msgs__msg__StateUint * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__StateUint__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__STATE_UINT__STRUCT_H_
