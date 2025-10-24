// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from mujoco_ros_msgs:msg/SolverParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__FUNCTIONS_H_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "mujoco_ros_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "mujoco_ros_msgs/msg/detail/solver_parameters__struct.h"

/// Initialize msg/SolverParameters message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__msg__SolverParameters
 * )) before or use
 * mujoco_ros_msgs__msg__SolverParameters__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__SolverParameters__init(mujoco_ros_msgs__msg__SolverParameters * msg);

/// Finalize msg/SolverParameters message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__msg__SolverParameters__fini(mujoco_ros_msgs__msg__SolverParameters * msg);

/// Create msg/SolverParameters message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__msg__SolverParameters__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__msg__SolverParameters *
mujoco_ros_msgs__msg__SolverParameters__create();

/// Destroy msg/SolverParameters message.
/**
 * It calls
 * mujoco_ros_msgs__msg__SolverParameters__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__msg__SolverParameters__destroy(mujoco_ros_msgs__msg__SolverParameters * msg);

/// Check for msg/SolverParameters message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__SolverParameters__are_equal(const mujoco_ros_msgs__msg__SolverParameters * lhs, const mujoco_ros_msgs__msg__SolverParameters * rhs);

/// Copy a msg/SolverParameters message.
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
mujoco_ros_msgs__msg__SolverParameters__copy(
  const mujoco_ros_msgs__msg__SolverParameters * input,
  mujoco_ros_msgs__msg__SolverParameters * output);

/// Initialize array of msg/SolverParameters messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__msg__SolverParameters__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__SolverParameters__Sequence__init(mujoco_ros_msgs__msg__SolverParameters__Sequence * array, size_t size);

/// Finalize array of msg/SolverParameters messages.
/**
 * It calls
 * mujoco_ros_msgs__msg__SolverParameters__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__msg__SolverParameters__Sequence__fini(mujoco_ros_msgs__msg__SolverParameters__Sequence * array);

/// Create array of msg/SolverParameters messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__msg__SolverParameters__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__msg__SolverParameters__Sequence *
mujoco_ros_msgs__msg__SolverParameters__Sequence__create(size_t size);

/// Destroy array of msg/SolverParameters messages.
/**
 * It calls
 * mujoco_ros_msgs__msg__SolverParameters__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__msg__SolverParameters__Sequence__destroy(mujoco_ros_msgs__msg__SolverParameters__Sequence * array);

/// Check for msg/SolverParameters message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__msg__SolverParameters__Sequence__are_equal(const mujoco_ros_msgs__msg__SolverParameters__Sequence * lhs, const mujoco_ros_msgs__msg__SolverParameters__Sequence * rhs);

/// Copy an array of msg/SolverParameters messages.
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
mujoco_ros_msgs__msg__SolverParameters__Sequence__copy(
  const mujoco_ros_msgs__msg__SolverParameters__Sequence * input,
  mujoco_ros_msgs__msg__SolverParameters__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__FUNCTIONS_H_
