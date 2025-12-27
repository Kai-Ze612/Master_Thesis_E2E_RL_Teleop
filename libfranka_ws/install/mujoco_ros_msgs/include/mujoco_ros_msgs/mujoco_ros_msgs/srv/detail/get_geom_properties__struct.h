// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:srv/GetGeomProperties.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GEOM_PROPERTIES__STRUCT_H_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GEOM_PROPERTIES__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'geom_name'
// Member 'admin_hash'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/GetGeomProperties in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__GetGeomProperties_Request
{
  rosidl_runtime_c__String geom_name;
  rosidl_runtime_c__String admin_hash;
} mujoco_ros_msgs__srv__GetGeomProperties_Request;

// Struct for a sequence of mujoco_ros_msgs__srv__GetGeomProperties_Request.
typedef struct mujoco_ros_msgs__srv__GetGeomProperties_Request__Sequence
{
  mujoco_ros_msgs__srv__GetGeomProperties_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__GetGeomProperties_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'properties'
#include "mujoco_ros_msgs/msg/detail/geom_properties__struct.h"
// Member 'status_message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/GetGeomProperties in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__srv__GetGeomProperties_Response
{
  mujoco_ros_msgs__msg__GeomProperties properties;
  bool success;
  rosidl_runtime_c__String status_message;
} mujoco_ros_msgs__srv__GetGeomProperties_Response;

// Struct for a sequence of mujoco_ros_msgs__srv__GetGeomProperties_Response.
typedef struct mujoco_ros_msgs__srv__GetGeomProperties_Response__Sequence
{
  mujoco_ros_msgs__srv__GetGeomProperties_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__srv__GetGeomProperties_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GEOM_PROPERTIES__STRUCT_H_
