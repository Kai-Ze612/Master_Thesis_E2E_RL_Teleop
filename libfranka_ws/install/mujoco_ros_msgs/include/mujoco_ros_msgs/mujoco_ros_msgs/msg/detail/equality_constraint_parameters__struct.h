// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/EqualityConstraintParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__STRUCT_H_

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
// Member 'class_param'
// Member 'element1'
// Member 'element2'
#include "rosidl_runtime_c/string.h"
// Member 'type'
#include "mujoco_ros_msgs/msg/detail/equality_constraint_type__struct.h"
// Member 'solver_parameters'
#include "mujoco_ros_msgs/msg/detail/solver_parameters__struct.h"
// Member 'anchor'
#include "geometry_msgs/msg/detail/vector3__struct.h"
// Member 'relpose'
#include "geometry_msgs/msg/detail/pose__struct.h"
// Member 'polycoef'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/EqualityConstraintParameters in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__msg__EqualityConstraintParameters
{
  rosidl_runtime_c__String name;
  mujoco_ros_msgs__msg__EqualityConstraintType type;
  mujoco_ros_msgs__msg__SolverParameters solver_parameters;
  bool active;
  rosidl_runtime_c__String class_param;
  rosidl_runtime_c__String element1;
  rosidl_runtime_c__String element2;
  double torquescale;
  geometry_msgs__msg__Vector3 anchor;
  geometry_msgs__msg__Pose relpose;
  rosidl_runtime_c__double__Sequence polycoef;
} mujoco_ros_msgs__msg__EqualityConstraintParameters;

// Struct for a sequence of mujoco_ros_msgs__msg__EqualityConstraintParameters.
typedef struct mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence
{
  mujoco_ros_msgs__msg__EqualityConstraintParameters * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__STRUCT_H_
