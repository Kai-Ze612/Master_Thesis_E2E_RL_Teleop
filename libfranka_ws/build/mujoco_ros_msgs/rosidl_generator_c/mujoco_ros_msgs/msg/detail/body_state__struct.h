// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/BodyState.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__BODY_STATE__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__BODY_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'name'
#include "rosidl_runtime_c/string.h"
// Member 'pose'
#include "geometry_msgs/msg/detail/pose_stamped__struct.h"
// Member 'twist'
#include "geometry_msgs/msg/detail/twist_stamped__struct.h"

/// Struct defined in msg/BodyState in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__msg__BodyState
{
  /// this can either be the body name or name of a geom belongig to the body.
  rosidl_runtime_c__String name;
  geometry_msgs__msg__PoseStamped pose;
  geometry_msgs__msg__TwistStamped twist;
  float mass;
} mujoco_ros_msgs__msg__BodyState;

// Struct for a sequence of mujoco_ros_msgs__msg__BodyState.
typedef struct mujoco_ros_msgs__msg__BodyState__Sequence
{
  mujoco_ros_msgs__msg__BodyState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__BodyState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__BODY_STATE__STRUCT_H_
