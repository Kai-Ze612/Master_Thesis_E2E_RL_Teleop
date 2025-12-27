// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from franka_msgs:msg/FrankaModel.idl
// generated code does not contain a copyright notice

#ifndef FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__STRUCT_H_
#define FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__STRUCT_H_

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

/// Struct defined in msg/FrankaModel in the package franka_msgs.
typedef struct franka_msgs__msg__FrankaModel
{
  std_msgs__msg__Header header;
  double coriolis[7];
  double ee_zero_jacobian[42];
  double ee_body_jacobian[42];
  double mass[49];
} franka_msgs__msg__FrankaModel;

// Struct for a sequence of franka_msgs__msg__FrankaModel.
typedef struct franka_msgs__msg__FrankaModel__Sequence
{
  franka_msgs__msg__FrankaModel * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} franka_msgs__msg__FrankaModel__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__STRUCT_H_
