// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/GeomType.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_TYPE__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_TYPE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'PLANE'.
enum
{
  mujoco_ros_msgs__msg__GeomType__PLANE = 0
};

/// Constant 'HFIELD'.
enum
{
  mujoco_ros_msgs__msg__GeomType__HFIELD = 1
};

/// Constant 'SPHERE'.
enum
{
  mujoco_ros_msgs__msg__GeomType__SPHERE = 2
};

/// Constant 'CAPSULE'.
enum
{
  mujoco_ros_msgs__msg__GeomType__CAPSULE = 3
};

/// Constant 'ELLIPSOID'.
enum
{
  mujoco_ros_msgs__msg__GeomType__ELLIPSOID = 4
};

/// Constant 'CYLINDER'.
enum
{
  mujoco_ros_msgs__msg__GeomType__CYLINDER = 5
};

/// Constant 'BOX'.
enum
{
  mujoco_ros_msgs__msg__GeomType__BOX = 6
};

/// Constant 'MESH'.
enum
{
  mujoco_ros_msgs__msg__GeomType__MESH = 7
};

/// Constant 'GEOM_NONE'.
enum
{
  mujoco_ros_msgs__msg__GeomType__GEOM_NONE = 1001
};

/// Struct defined in msg/GeomType in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__msg__GeomType
{
  uint16_t value;
} mujoco_ros_msgs__msg__GeomType;

// Struct for a sequence of mujoco_ros_msgs__msg__GeomType.
typedef struct mujoco_ros_msgs__msg__GeomType__Sequence
{
  mujoco_ros_msgs__msg__GeomType * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__GeomType__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_TYPE__STRUCT_H_
