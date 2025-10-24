// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from multi_mode_control_msgs:msg/ContactThresholds.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__STRUCT_H_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'tau_max'
// Member 'tau_min'
#include "multi_mode_control_msgs/msg/detail/joint_array__struct.h"

/// Struct defined in msg/ContactThresholds in the package multi_mode_control_msgs.
typedef struct multi_mode_control_msgs__msg__ContactThresholds
{
  double f_max[6];
  double f_min[6];
  multi_mode_control_msgs__msg__JointArray__Sequence tau_max;
  multi_mode_control_msgs__msg__JointArray__Sequence tau_min;
  double f_abs_max;
  double m_abs_max;
  bool abs_mode;
  bool directional_mode;
  bool torque_mode;
} multi_mode_control_msgs__msg__ContactThresholds;

// Struct for a sequence of multi_mode_control_msgs__msg__ContactThresholds.
typedef struct multi_mode_control_msgs__msg__ContactThresholds__Sequence
{
  multi_mode_control_msgs__msg__ContactThresholds * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} multi_mode_control_msgs__msg__ContactThresholds__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__STRUCT_H_
