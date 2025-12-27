// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:srv/GetStateUint.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_STATE_UINT__STRUCT_H_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_STATE_UINT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetStateUint in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__GetStateUint_Request
{
  uint8_t structure_needs_at_least_one_member;
} mujoco_ros_msgs__srv__GetStateUint_Request;

// Struct for a sequence of mujoco_ros_msgs__srv__GetStateUint_Request.
typedef struct mujoco_ros_msgs__srv__GetStateUint_Request__Sequence
{
  mujoco_ros_msgs__srv__GetStateUint_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__GetStateUint_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'state'
#include "mujoco_ros_msgs/msg/detail/state_uint__struct.h"

/// Struct defined in srv/GetStateUint in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__GetStateUint_Response
{
  mujoco_ros_msgs__msg__StateUint state;
} mujoco_ros_msgs__srv__GetStateUint_Response;

// Struct for a sequence of mujoco_ros_msgs__srv__GetStateUint_Response.
typedef struct mujoco_ros_msgs__srv__GetStateUint_Response__Sequence
{
  mujoco_ros_msgs__srv__GetStateUint_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__GetStateUint_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_STATE_UINT__STRUCT_H_
