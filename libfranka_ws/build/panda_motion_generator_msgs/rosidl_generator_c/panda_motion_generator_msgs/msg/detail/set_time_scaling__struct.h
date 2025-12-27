// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from panda_motion_generator_msgs:msg/SetTimeScaling.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__STRUCT_H_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/SetTimeScaling in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__msg__SetTimeScaling
{
  int32_t steps;
  double scaling;
} panda_motion_generator_msgs__msg__SetTimeScaling;

// Struct for a sequence of panda_motion_generator_msgs__msg__SetTimeScaling.
typedef struct panda_motion_generator_msgs__msg__SetTimeScaling__Sequence
{
  panda_motion_generator_msgs__msg__SetTimeScaling * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__msg__SetTimeScaling__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__STRUCT_H_
