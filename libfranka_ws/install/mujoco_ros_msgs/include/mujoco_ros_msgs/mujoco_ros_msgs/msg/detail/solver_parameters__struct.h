// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/SolverParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/SolverParameters in the package mujoco_ros_msgs.
/**
  * solimp parameters
 */
typedef struct mujoco_ros_msgs__msg__SolverParameters
{
  double dmin;
  double dmax;
  double width;
  double midpoint;
  double power;
  /// solref parameters
  double timeconst;
  double dampratio;
} mujoco_ros_msgs__msg__SolverParameters;

// Struct for a sequence of mujoco_ros_msgs__msg__SolverParameters.
typedef struct mujoco_ros_msgs__msg__SolverParameters__Sequence
{
  mujoco_ros_msgs__msg__SolverParameters * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__SolverParameters__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__STRUCT_H_
