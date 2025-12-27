// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/ScalarStamped.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SCALAR_STAMPED__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SCALAR_STAMPED__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/ScalarStamped in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__msg__ScalarStamped
{
  std_msgs__msg__Header header;
  double value;
} mujoco_ros_msgs__msg__ScalarStamped;

// Struct for a sequence of mujoco_ros_msgs__msg__ScalarStamped.
typedef struct mujoco_ros_msgs__msg__ScalarStamped__Sequence
{
  mujoco_ros_msgs__msg__ScalarStamped * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__ScalarStamped__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SCALAR_STAMPED__STRUCT_H_
