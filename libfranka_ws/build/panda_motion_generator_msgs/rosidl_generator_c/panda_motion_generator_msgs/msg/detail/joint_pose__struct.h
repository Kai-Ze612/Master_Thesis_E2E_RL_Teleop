// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from panda_motion_generator_msgs:msg/JointPose.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__STRUCT_H_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'q'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/JointPose in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__msg__JointPose
{
  rosidl_runtime_c__double__Sequence q;
} panda_motion_generator_msgs__msg__JointPose;

// Struct for a sequence of panda_motion_generator_msgs__msg__JointPose.
typedef struct panda_motion_generator_msgs__msg__JointPose__Sequence
{
  panda_motion_generator_msgs__msg__JointPose * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__msg__JointPose__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__STRUCT_H_
