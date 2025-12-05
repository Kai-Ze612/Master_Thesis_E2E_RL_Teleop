// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:srv/SetMocapState.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__SET_MOCAP_STATE__STRUCT_H_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__SET_MOCAP_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'mocap_state'
#include "mujoco_ros_msgs/msg/detail/mocap_state__struct.h"

/// Struct defined in srv/SetMocapState in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__SetMocapState_Request
{
  mujoco_ros_msgs__msg__MocapState mocap_state;
} mujoco_ros_msgs__srv__SetMocapState_Request;

// Struct for a sequence of mujoco_ros_msgs__srv__SetMocapState_Request.
typedef struct mujoco_ros_msgs__srv__SetMocapState_Request__Sequence
{
  mujoco_ros_msgs__srv__SetMocapState_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__SetMocapState_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetMocapState in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__SetMocapState_Response
{
  bool success;
} mujoco_ros_msgs__srv__SetMocapState_Response;

// Struct for a sequence of mujoco_ros_msgs__srv__SetMocapState_Response.
typedef struct mujoco_ros_msgs__srv__SetMocapState_Response__Sequence
{
  mujoco_ros_msgs__srv__SetMocapState_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__SetMocapState_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__SET_MOCAP_STATE__STRUCT_H_
