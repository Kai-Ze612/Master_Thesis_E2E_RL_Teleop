// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:msg/SensorNoiseModel.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__STRUCT_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'sensor_name'
#include "rosidl_runtime_c/string.h"
// Member 'mean'
// Member 'std'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/SensorNoiseModel in the package mujoco_ros_msgs.
/**
  * Set the noise model of a sensor defining mean and standard deviation for each dimension
  * For quaternion sensors noise is calculated in euler angles (rad), converted to a quaternion and then applied. Thus only three mean/std pairs are required!
 */
typedef struct mujoco_ros_msgs__msg__SensorNoiseModel
{
  rosidl_runtime_c__String sensor_name;
  rosidl_runtime_c__double__Sequence mean;
  rosidl_runtime_c__double__Sequence std;
  uint8_t set_flag;
} mujoco_ros_msgs__msg__SensorNoiseModel;

// Struct for a sequence of mujoco_ros_msgs__msg__SensorNoiseModel.
typedef struct mujoco_ros_msgs__msg__SensorNoiseModel__Sequence
{
  mujoco_ros_msgs__msg__SensorNoiseModel * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__msg__SensorNoiseModel__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__STRUCT_H_
