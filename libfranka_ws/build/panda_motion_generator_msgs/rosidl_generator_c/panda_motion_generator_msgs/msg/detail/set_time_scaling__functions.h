// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from panda_motion_generator_msgs:msg/SetTimeScaling.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__FUNCTIONS_H_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "panda_motion_generator_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "panda_motion_generator_msgs/msg/detail/set_time_scaling__struct.h"

/// Initialize msg/SetTimeScaling message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * panda_motion_generator_msgs__msg__SetTimeScaling
 * )) before or use
 * panda_motion_generator_msgs__msg__SetTimeScaling__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__msg__SetTimeScaling__init(panda_motion_generator_msgs__msg__SetTimeScaling * msg);

/// Finalize msg/SetTimeScaling message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__msg__SetTimeScaling__fini(panda_motion_generator_msgs__msg__SetTimeScaling * msg);

/// Create msg/SetTimeScaling message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * panda_motion_generator_msgs__msg__SetTimeScaling__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__msg__SetTimeScaling *
panda_motion_generator_msgs__msg__SetTimeScaling__create();

/// Destroy msg/SetTimeScaling message.
/**
 * It calls
 * panda_motion_generator_msgs__msg__SetTimeScaling__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__msg__SetTimeScaling__destroy(panda_motion_generator_msgs__msg__SetTimeScaling * msg);

/// Check for msg/SetTimeScaling message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__msg__SetTimeScaling__are_equal(const panda_motion_generator_msgs__msg__SetTimeScaling * lhs, const panda_motion_generator_msgs__msg__SetTimeScaling * rhs);

/// Copy a msg/SetTimeScaling message.
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
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__msg__SetTimeScaling__copy(
  const panda_motion_generator_msgs__msg__SetTimeScaling * input,
  panda_motion_generator_msgs__msg__SetTimeScaling * output);

/// Initialize array of msg/SetTimeScaling messages.
/**
 * It allocates the memory for the number of elements and calls
 * panda_motion_generator_msgs__msg__SetTimeScaling__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__init(panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * array, size_t size);

/// Finalize array of msg/SetTimeScaling messages.
/**
 * It calls
 * panda_motion_generator_msgs__msg__SetTimeScaling__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__fini(panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * array);

/// Create array of msg/SetTimeScaling messages.
/**
 * It allocates the memory for the array and calls
 * panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence *
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__create(size_t size);

/// Destroy array of msg/SetTimeScaling messages.
/**
 * It calls
 * panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__destroy(panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * array);

/// Check for msg/SetTimeScaling message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__are_equal(const panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * lhs, const panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * rhs);

/// Copy an array of msg/SetTimeScaling messages.
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
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__copy(
  const panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * input,
  panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__FUNCTIONS_H_
