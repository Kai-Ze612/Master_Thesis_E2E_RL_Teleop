// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from multi_mode_control_msgs:msg/Controller.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__FUNCTIONS_H_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "multi_mode_control_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "multi_mode_control_msgs/msg/detail/controller__struct.h"

/// Initialize msg/Controller message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * multi_mode_control_msgs__msg__Controller
 * )) before or use
 * multi_mode_control_msgs__msg__Controller__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__msg__Controller__init(multi_mode_control_msgs__msg__Controller * msg);

/// Finalize msg/Controller message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__msg__Controller__fini(multi_mode_control_msgs__msg__Controller * msg);

/// Create msg/Controller message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * multi_mode_control_msgs__msg__Controller__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
multi_mode_control_msgs__msg__Controller *
multi_mode_control_msgs__msg__Controller__create();

/// Destroy msg/Controller message.
/**
 * It calls
 * multi_mode_control_msgs__msg__Controller__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__msg__Controller__destroy(multi_mode_control_msgs__msg__Controller * msg);

/// Check for msg/Controller message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__msg__Controller__are_equal(const multi_mode_control_msgs__msg__Controller * lhs, const multi_mode_control_msgs__msg__Controller * rhs);

/// Copy a msg/Controller message.
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
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__msg__Controller__copy(
  const multi_mode_control_msgs__msg__Controller * input,
  multi_mode_control_msgs__msg__Controller * output);

/// Initialize array of msg/Controller messages.
/**
 * It allocates the memory for the number of elements and calls
 * multi_mode_control_msgs__msg__Controller__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__msg__Controller__Sequence__init(multi_mode_control_msgs__msg__Controller__Sequence * array, size_t size);

/// Finalize array of msg/Controller messages.
/**
 * It calls
 * multi_mode_control_msgs__msg__Controller__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__msg__Controller__Sequence__fini(multi_mode_control_msgs__msg__Controller__Sequence * array);

/// Create array of msg/Controller messages.
/**
 * It allocates the memory for the array and calls
 * multi_mode_control_msgs__msg__Controller__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
multi_mode_control_msgs__msg__Controller__Sequence *
multi_mode_control_msgs__msg__Controller__Sequence__create(size_t size);

/// Destroy array of msg/Controller messages.
/**
 * It calls
 * multi_mode_control_msgs__msg__Controller__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__msg__Controller__Sequence__destroy(multi_mode_control_msgs__msg__Controller__Sequence * array);

/// Check for msg/Controller message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__msg__Controller__Sequence__are_equal(const multi_mode_control_msgs__msg__Controller__Sequence * lhs, const multi_mode_control_msgs__msg__Controller__Sequence * rhs);

/// Copy an array of msg/Controller messages.
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
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__msg__Controller__Sequence__copy(
  const multi_mode_control_msgs__msg__Controller__Sequence * input,
  multi_mode_control_msgs__msg__Controller__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__FUNCTIONS_H_
