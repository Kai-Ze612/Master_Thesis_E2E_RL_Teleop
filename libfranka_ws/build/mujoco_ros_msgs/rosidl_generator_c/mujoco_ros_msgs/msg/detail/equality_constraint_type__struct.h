// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/EqualityConstraintType.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_TYPE__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_TYPE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'CONNECT'.
enum
{
  mujoco_ros_msgs__msg__EqualityConstraintType__CONNECT = 0
};

/// Constant 'WELD'.
enum
{
  mujoco_ros_msgs__msg__EqualityConstraintType__WELD = 1
};

/// Constant 'JOINT'.
enum
{
  mujoco_ros_msgs__msg__EqualityConstraintType__JOINT = 2
};

/// Constant 'TENDON'.
enum
{
  mujoco_ros_msgs__msg__EqualityConstraintType__TENDON = 3
};

/// Struct defined in msg/EqualityConstraintType in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__msg__EqualityConstraintType
{
  uint16_t value;
} mujoco_ros_msgs__msg__EqualityConstraintType;

// Struct for a sequence of mujoco_ros_msgs__msg__EqualityConstraintType.
typedef struct mujoco_ros_msgs__msg__EqualityConstraintType__Sequence
{
  mujoco_ros_msgs__msg__EqualityConstraintType * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__EqualityConstraintType__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_TYPE__STRUCT_H_
