// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from multi_mode_control_msgs:msg/JointGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__JOINT_GOAL__STRUCT_H_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__JOINT_GOAL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/JointGoal in the package multi_mode_control_msgs.
typedef struct multi_mode_control_msgs__msg__JointGoal
{
  double q[7];
  double qd[7];
} multi_mode_control_msgs__msg__JointGoal;

// Struct for a sequence of multi_mode_control_msgs__msg__JointGoal.
typedef struct multi_mode_control_msgs__msg__JointGoal__Sequence
{
  multi_mode_control_msgs__msg__JointGoal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} multi_mode_control_msgs__msg__JointGoal__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__JOINT_GOAL__STRUCT_H_
