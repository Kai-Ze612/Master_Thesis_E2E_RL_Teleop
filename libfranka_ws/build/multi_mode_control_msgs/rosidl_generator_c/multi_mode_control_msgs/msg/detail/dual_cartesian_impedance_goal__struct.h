// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from multi_mode_control_msgs:msg/DualCartesianImpedanceGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__STRUCT_H_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'l_pose'
// Member 'r_pose'
#include "geometry_msgs/msg/detail/pose__struct.h"

/// Struct defined in msg/DualCartesianImpedanceGoal in the package multi_mode_control_msgs.
typedef struct multi_mode_control_msgs__msg__DualCartesianImpedanceGoal
{
  geometry_msgs__msg__Pose l_pose;
  geometry_msgs__msg__Pose r_pose;
  double l_q_n[7];
  double r_q_n[7];
} multi_mode_control_msgs__msg__DualCartesianImpedanceGoal;

// Struct for a sequence of multi_mode_control_msgs__msg__DualCartesianImpedanceGoal.
typedef struct multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence
{
  multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__STRUCT_H_
