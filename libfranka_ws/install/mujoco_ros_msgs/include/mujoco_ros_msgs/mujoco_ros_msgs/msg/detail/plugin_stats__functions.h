// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from mujoco_ros_msgs:msg/PluginStats.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__FUNCTIONS_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "mujoco_ros_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "mujoco_ros_msgs/msg/detail/plugin_stats__struct.h"

/// Initialize msg/PluginStats message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__msg__PluginStats
 * )) before or use
 * mujoco_ros_msgs__msg__PluginStats__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__PluginStats__init(mujoco_ros_msgs__msg__PluginStats * msg);

/// Finalize msg/PluginStats message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__msg__PluginStats__fini(mujoco_ros_msgs__msg__PluginStats * msg);

/// Create msg/PluginStats message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__msg__PluginStats__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__msg__PluginStats *
mujoco_ros_msgs__msg__PluginStats__create();

/// Destroy msg/PluginStats message.
/**
 * It calls
 * mujoco_ros_msgs__msg__PluginStats__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__msg__PluginStats__destroy(mujoco_ros_msgs__msg__PluginStats * msg);

/// Check for msg/PluginStats message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__PluginStats__are_equal(const mujoco_ros_msgs__msg__PluginStats * lhs, const mujoco_ros_msgs__msg__PluginStats * rhs);

/// Copy a msg/PluginStats message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__PluginStats__copy(
  const mujoco_ros_msgs__msg__PluginStats * input,
  mujoco_ros_msgs__msg__PluginStats * output);

/// Initialize array of msg/PluginStats messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__msg__PluginStats__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__PluginStats__Sequence__init(mujoco_ros_msgs__msg__PluginStats__Sequence * array, size_t size);

/// Finalize array of msg/PluginStats messages.
/**
 * It calls
 * mujoco_ros_msgs__msg__PluginStats__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__msg__PluginStats__Sequence__fini(mujoco_ros_msgs__msg__PluginStats__Sequence * array);

/// Create array of msg/PluginStats messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__msg__PluginStats__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__msg__PluginStats__Sequence *
mujoco_ros_msgs__msg__PluginStats__Sequence__create(size_t size);

/// Destroy array of msg/PluginStats messages.
/**
 * It calls
 * mujoco_ros_msgs__msg__PluginStats__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__msg__PluginStats__Sequence__destroy(mujoco_ros_msgs__msg__PluginStats__Sequence * array);

/// Check for msg/PluginStats message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__PluginStats__Sequence__are_equal(const mujoco_ros_msgs__msg__PluginStats__Sequence * lhs, const mujoco_ros_msgs__msg__PluginStats__Sequence * rhs);

/// Copy an array of msg/PluginStats messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__PluginStats__Sequence__copy(
  const mujoco_ros_msgs__msg__PluginStats__Sequence * input,
  mujoco_ros_msgs__msg__PluginStats__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__FUNCTIONS_H_
