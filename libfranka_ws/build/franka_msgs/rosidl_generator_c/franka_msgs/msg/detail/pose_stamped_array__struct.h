// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from franka_msgs:msg/PoseStampedArray.idl
// generated code does not contain a copyright notice

#ifndef FRANKA_MSGS__MSG__DETAIL__POSE_STAMPED_ARRAY__STRUCT_H_
#define FRANKA_MSGS__MSG__DETAIL__POSE_STAMPED_ARRAY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'stampedposes'
#include "geometry_msgs/msg/detail/pose_stamped__struct.h"

/// Struct defined in msg/PoseStampedArray in the package franka_msgs.
typedef struct franka_msgs__msg__PoseStampedArray
{
  geometry_msgs__msg__PoseStamped__Sequence stampedposes;
} franka_msgs__msg__PoseStampedArray;

// Struct for a sequence of franka_msgs__msg__PoseStampedArray.
typedef struct franka_msgs__msg__PoseStampedArray__Sequence
{
  franka_msgs__msg__PoseStampedArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} franka_msgs__msg__PoseStampedArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FRANKA_MSGS__MSG__DETAIL__POSE_STAMPED_ARRAY__STRUCT_H_
