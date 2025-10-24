// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:srv/GetGravity.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GRAVITY__STRUCT_H_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GRAVITY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'admin_hash'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/GetGravity in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__GetGravity_Request
{
  rosidl_runtime_c__String admin_hash;
} mujoco_ros_msgs__srv__GetGravity_Request;

// Struct for a sequence of mujoco_ros_msgs__srv__GetGravity_Request.
typedef struct mujoco_ros_msgs__srv__GetGravity_Request__Sequence
{
  mujoco_ros_msgs__srv__GetGravity_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__GetGravity_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'status_message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/GetGravity in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__GetGravity_Response
{
  double gravity[3];
  bool success;
  rosidl_runtime_c__String status_message;
} mujoco_ros_msgs__srv__GetGravity_Response;

// Struct for a sequence of mujoco_ros_msgs__srv__GetGravity_Response.
typedef struct mujoco_ros_msgs__srv__GetGravity_Response__Sequence
{
  mujoco_ros_msgs__srv__GetGravity_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__GetGravity_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GRAVITY__STRUCT_H_
