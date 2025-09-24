// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/GeomProperties.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__STRUCT_H_

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
#include "rosidl_runtime_c/string.h"
// Member 'type'
#include "mujoco_ros_msgs/msg/detail/geom_type__struct.h"

/// Struct defined in msg/GeomProperties in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__msg__GeomProperties
{
  rosidl_runtime_c__String name;
  mujoco_ros_msgs__msg__GeomType type;
  /// total mass of the body this geom belongs to
  float body_mass;
  float size_0;
  float size_1;
  float size_2;
  float friction_slide;
  float friction_spin;
  float friction_roll;
} mujoco_ros_msgs__msg__GeomProperties;

// Struct for a sequence of mujoco_ros_msgs__msg__GeomProperties.
typedef struct mujoco_ros_msgs__msg__GeomProperties__Sequence
{
  mujoco_ros_msgs__msg__GeomProperties * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__GeomProperties__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__STRUCT_H_
