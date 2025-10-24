// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from panda_motion_generator_msgs:msg/SimpleActionResult.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__STRUCT_H_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'SUCCESS'.
enum
{
  panda_motion_generator_msgs__msg__SimpleActionResult__SUCCESS = 0l
};

/// Constant 'FAILURE'.
enum
{
  panda_motion_generator_msgs__msg__SimpleActionResult__FAILURE = 1l
};

/// Constant 'REJECTED'.
enum
{
  panda_motion_generator_msgs__msg__SimpleActionResult__REJECTED = 2l
};

/// Constant 'ABORTED'.
enum
{
  panda_motion_generator_msgs__msg__SimpleActionResult__ABORTED = 3l
};

/// Constant 'PREEMPTED'.
enum
{
  panda_motion_generator_msgs__msg__SimpleActionResult__PREEMPTED = 4l
};

// Include directives for member types
// Member 'error'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/SimpleActionResult in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__msg__SimpleActionResult
{
  int32_t state;
  rosidl_runtime_c__String error;
} panda_motion_generator_msgs__msg__SimpleActionResult;

// Struct for a sequence of panda_motion_generator_msgs__msg__SimpleActionResult.
typedef struct panda_motion_generator_msgs__msg__SimpleActionResult__Sequence
{
  panda_motion_generator_msgs__msg__SimpleActionResult * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__msg__SimpleActionResult__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__STRUCT_H_
