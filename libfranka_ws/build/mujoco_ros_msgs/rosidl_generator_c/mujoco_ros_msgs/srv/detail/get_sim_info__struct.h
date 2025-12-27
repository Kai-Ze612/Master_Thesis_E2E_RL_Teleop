// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:srv/GetSimInfo.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_SIM_INFO__STRUCT_H_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_SIM_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetSimInfo in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__GetSimInfo_Request
{
  uint8_t structure_needs_at_least_one_member;
} mujoco_ros_msgs__srv__GetSimInfo_Request;

// Struct for a sequence of mujoco_ros_msgs__srv__GetSimInfo_Request.
typedef struct mujoco_ros_msgs__srv__GetSimInfo_Request__Sequence
{
  mujoco_ros_msgs__srv__GetSimInfo_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__GetSimInfo_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'state'
#include "mujoco_ros_msgs/msg/detail/sim_info__struct.h"

/// Struct defined in srv/GetSimInfo in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__GetSimInfo_Response
{
  mujoco_ros_msgs__msg__SimInfo state;
} mujoco_ros_msgs__srv__GetSimInfo_Response;

// Struct for a sequence of mujoco_ros_msgs__srv__GetSimInfo_Response.
typedef struct mujoco_ros_msgs__srv__GetSimInfo_Response__Sequence
{
  mujoco_ros_msgs__srv__GetSimInfo_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__GetSimInfo_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_SIM_INFO__STRUCT_H_
