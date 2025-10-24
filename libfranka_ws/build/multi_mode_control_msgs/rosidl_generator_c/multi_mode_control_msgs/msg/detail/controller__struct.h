// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from multi_mode_control_msgs:msg/Controller.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__STRUCT_H_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__STRUCT_H_

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
// Member 'resources'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/Controller in the package multi_mode_control_msgs.
typedef struct multi_mode_control_msgs__msg__Controller
{
  rosidl_runtime_c__String name;
  rosidl_runtime_c__String__Sequence resources;
} multi_mode_control_msgs__msg__Controller;

// Struct for a sequence of multi_mode_control_msgs__msg__Controller.
typedef struct multi_mode_control_msgs__msg__Controller__Sequence
{
  multi_mode_control_msgs__msg__Controller * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} multi_mode_control_msgs__msg__Controller__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__STRUCT_H_
