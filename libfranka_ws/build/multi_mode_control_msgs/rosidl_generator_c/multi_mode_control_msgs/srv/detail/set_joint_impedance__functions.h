// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from multi_mode_control_msgs:srv/SetJointImpedance.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_JOINT_IMPEDANCE__FUNCTIONS_H_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_JOINT_IMPEDANCE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "multi_mode_control_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "multi_mode_control_msgs/srv/detail/set_joint_impedance__struct.h"

/// Initialize srv/SetJointImpedance message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * multi_mode_control_msgs__srv__SetJointImpedance_Request
 * )) before or use
 * multi_mode_control_msgs__srv__SetJointImpedance_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__srv__SetJointImpedance_Request__init(multi_mode_control_msgs__srv__SetJointImpedance_Request * msg);

/// Finalize srv/SetJointImpedance message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__srv__SetJointImpedance_Request__fini(multi_mode_control_msgs__srv__SetJointImpedance_Request * msg);

/// Create srv/SetJointImpedance message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
multi_mode_control_msgs__srv__SetJointImpedance_Request *
multi_mode_control_msgs__srv__SetJointImpedance_Request__create();

/// Destroy srv/SetJointImpedance message.
/**
 * It calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__srv__SetJointImpedance_Request__destroy(multi_mode_control_msgs__srv__SetJointImpedance_Request * msg);

/// Check for srv/SetJointImpedance message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__srv__SetJointImpedance_Request__are_equal(const multi_mode_control_msgs__srv__SetJointImpedance_Request * lhs, const multi_mode_control_msgs__srv__SetJointImpedance_Request * rhs);

/// Copy a srv/SetJointImpedance message.
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
multi_mode_control_msgs__srv__SetJointImpedance_Request__copy(
  const multi_mode_control_msgs__srv__SetJointImpedance_Request * input,
  multi_mode_control_msgs__srv__SetJointImpedance_Request * output);

/// Initialize array of srv/SetJointImpedance messages.
/**
 * It allocates the memory for the number of elements and calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence__init(multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence * array, size_t size);

/// Finalize array of srv/SetJointImpedance messages.
/**
 * It calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence__fini(multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence * array);

/// Create array of srv/SetJointImpedance messages.
/**
 * It allocates the memory for the array and calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence *
multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence__create(size_t size);

/// Destroy array of srv/SetJointImpedance messages.
/**
 * It calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence__destroy(multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence * array);

/// Check for srv/SetJointImpedance message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence__are_equal(const multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence * lhs, const multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence * rhs);

/// Copy an array of srv/SetJointImpedance messages.
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
multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence__copy(
  const multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence * input,
  multi_mode_control_msgs__srv__SetJointImpedance_Request__Sequence * output);

/// Initialize srv/SetJointImpedance message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * multi_mode_control_msgs__srv__SetJointImpedance_Response
 * )) before or use
 * multi_mode_control_msgs__srv__SetJointImpedance_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__srv__SetJointImpedance_Response__init(multi_mode_control_msgs__srv__SetJointImpedance_Response * msg);

/// Finalize srv/SetJointImpedance message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__srv__SetJointImpedance_Response__fini(multi_mode_control_msgs__srv__SetJointImpedance_Response * msg);

/// Create srv/SetJointImpedance message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
multi_mode_control_msgs__srv__SetJointImpedance_Response *
multi_mode_control_msgs__srv__SetJointImpedance_Response__create();

/// Destroy srv/SetJointImpedance message.
/**
 * It calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__srv__SetJointImpedance_Response__destroy(multi_mode_control_msgs__srv__SetJointImpedance_Response * msg);

/// Check for srv/SetJointImpedance message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__srv__SetJointImpedance_Response__are_equal(const multi_mode_control_msgs__srv__SetJointImpedance_Response * lhs, const multi_mode_control_msgs__srv__SetJointImpedance_Response * rhs);

/// Copy a srv/SetJointImpedance message.
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
multi_mode_control_msgs__srv__SetJointImpedance_Response__copy(
  const multi_mode_control_msgs__srv__SetJointImpedance_Response * input,
  multi_mode_control_msgs__srv__SetJointImpedance_Response * output);

/// Initialize array of srv/SetJointImpedance messages.
/**
 * It allocates the memory for the number of elements and calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence__init(multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence * array, size_t size);

/// Finalize array of srv/SetJointImpedance messages.
/**
 * It calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence__fini(multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence * array);

/// Create array of srv/SetJointImpedance messages.
/**
 * It allocates the memory for the array and calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence *
multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence__create(size_t size);

/// Destroy array of srv/SetJointImpedance messages.
/**
 * It calls
 * multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
void
multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence__destroy(multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence * array);

/// Check for srv/SetJointImpedance message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_multi_mode_control_msgs
bool
multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence__are_equal(const multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence * lhs, const multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence * rhs);

/// Copy an array of srv/SetJointImpedance messages.
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
multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence__copy(
  const multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence * input,
  multi_mode_control_msgs__srv__SetJointImpedance_Response__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_JOINT_IMPEDANCE__FUNCTIONS_H_
