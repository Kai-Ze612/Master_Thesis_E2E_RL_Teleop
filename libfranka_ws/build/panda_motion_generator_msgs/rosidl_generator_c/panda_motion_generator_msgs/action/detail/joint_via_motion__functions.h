// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from panda_motion_generator_msgs:action/JointViaMotion.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__JOINT_VIA_MOTION__FUNCTIONS_H_
#define PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__JOINT_VIA_MOTION__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "panda_motion_generator_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "panda_motion_generator_msgs/action/detail/joint_via_motion__struct.h"

/// Initialize action/JointViaMotion message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * panda_motion_generator_msgs__action__JointViaMotion_Goal
 * )) before or use
 * panda_motion_generator_msgs__action__JointViaMotion_Goal__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__init(panda_motion_generator_msgs__action__JointViaMotion_Goal * msg);

/// Finalize action/JointViaMotion message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Goal__fini(panda_motion_generator_msgs__action__JointViaMotion_Goal * msg);

/// Create action/JointViaMotion message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * panda_motion_generator_msgs__action__JointViaMotion_Goal__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_Goal *
panda_motion_generator_msgs__action__JointViaMotion_Goal__create();

/// Destroy action/JointViaMotion message.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_Goal__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Goal__destroy(panda_motion_generator_msgs__action__JointViaMotion_Goal * msg);

/// Check for action/JointViaMotion message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Goal * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Goal * rhs);

/// Copy a action/JointViaMotion message.
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
panda_motion_generator_msgs__action__JointViaMotion_Goal__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Goal * input,
  panda_motion_generator_msgs__action__JointViaMotion_Goal * output);

/// Initialize array of action/JointViaMotion messages.
/**
 * It allocates the memory for the number of elements and calls
 * panda_motion_generator_msgs__action__JointViaMotion_Goal__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * array, size_t size);

/// Finalize array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_Goal__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * array);

/// Create array of action/JointViaMotion messages.
/**
 * It allocates the memory for the array and calls
 * panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__create(size_t size);

/// Destroy array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * array);

/// Check for action/JointViaMotion message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * rhs);

/// Copy an array of action/JointViaMotion messages.
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
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * output);

/// Initialize action/JointViaMotion message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * panda_motion_generator_msgs__action__JointViaMotion_Result
 * )) before or use
 * panda_motion_generator_msgs__action__JointViaMotion_Result__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Result__init(panda_motion_generator_msgs__action__JointViaMotion_Result * msg);

/// Finalize action/JointViaMotion message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Result__fini(panda_motion_generator_msgs__action__JointViaMotion_Result * msg);

/// Create action/JointViaMotion message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * panda_motion_generator_msgs__action__JointViaMotion_Result__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_Result *
panda_motion_generator_msgs__action__JointViaMotion_Result__create();

/// Destroy action/JointViaMotion message.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_Result__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Result__destroy(panda_motion_generator_msgs__action__JointViaMotion_Result * msg);

/// Check for action/JointViaMotion message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Result__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Result * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Result * rhs);

/// Copy a action/JointViaMotion message.
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
panda_motion_generator_msgs__action__JointViaMotion_Result__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Result * input,
  panda_motion_generator_msgs__action__JointViaMotion_Result * output);

/// Initialize array of action/JointViaMotion messages.
/**
 * It allocates the memory for the number of elements and calls
 * panda_motion_generator_msgs__action__JointViaMotion_Result__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * array, size_t size);

/// Finalize array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_Result__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * array);

/// Create array of action/JointViaMotion messages.
/**
 * It allocates the memory for the array and calls
 * panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__create(size_t size);

/// Destroy array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * array);

/// Check for action/JointViaMotion message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * rhs);

/// Copy an array of action/JointViaMotion messages.
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
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * output);

/// Initialize action/JointViaMotion message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * panda_motion_generator_msgs__action__JointViaMotion_Feedback
 * )) before or use
 * panda_motion_generator_msgs__action__JointViaMotion_Feedback__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__init(panda_motion_generator_msgs__action__JointViaMotion_Feedback * msg);

/// Finalize action/JointViaMotion message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Feedback__fini(panda_motion_generator_msgs__action__JointViaMotion_Feedback * msg);

/// Create action/JointViaMotion message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * panda_motion_generator_msgs__action__JointViaMotion_Feedback__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_Feedback *
panda_motion_generator_msgs__action__JointViaMotion_Feedback__create();

/// Destroy action/JointViaMotion message.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_Feedback__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Feedback__destroy(panda_motion_generator_msgs__action__JointViaMotion_Feedback * msg);

/// Check for action/JointViaMotion message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Feedback * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Feedback * rhs);

/// Copy a action/JointViaMotion message.
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
panda_motion_generator_msgs__action__JointViaMotion_Feedback__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Feedback * input,
  panda_motion_generator_msgs__action__JointViaMotion_Feedback * output);

/// Initialize array of action/JointViaMotion messages.
/**
 * It allocates the memory for the number of elements and calls
 * panda_motion_generator_msgs__action__JointViaMotion_Feedback__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * array, size_t size);

/// Finalize array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_Feedback__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * array);

/// Create array of action/JointViaMotion messages.
/**
 * It allocates the memory for the array and calls
 * panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__create(size_t size);

/// Destroy array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * array);

/// Check for action/JointViaMotion message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * rhs);

/// Copy an array of action/JointViaMotion messages.
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
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * output);

/// Initialize action/JointViaMotion message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request
 * )) before or use
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__init(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * msg);

/// Finalize action/JointViaMotion message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * msg);

/// Create action/JointViaMotion message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request *
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__create();

/// Destroy action/JointViaMotion message.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__destroy(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * msg);

/// Check for action/JointViaMotion message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * lhs, const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * rhs);

/// Copy a action/JointViaMotion message.
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
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * input,
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * output);

/// Initialize array of action/JointViaMotion messages.
/**
 * It allocates the memory for the number of elements and calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * array, size_t size);

/// Finalize array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * array);

/// Create array of action/JointViaMotion messages.
/**
 * It allocates the memory for the array and calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__create(size_t size);

/// Destroy array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * array);

/// Check for action/JointViaMotion message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * rhs);

/// Copy an array of action/JointViaMotion messages.
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
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * output);

/// Initialize action/JointViaMotion message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response
 * )) before or use
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__init(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * msg);

/// Finalize action/JointViaMotion message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__fini(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * msg);

/// Create action/JointViaMotion message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response *
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__create();

/// Destroy action/JointViaMotion message.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__destroy(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * msg);

/// Check for action/JointViaMotion message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * lhs, const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * rhs);

/// Copy a action/JointViaMotion message.
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
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * input,
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * output);

/// Initialize array of action/JointViaMotion messages.
/**
 * It allocates the memory for the number of elements and calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * array, size_t size);

/// Finalize array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * array);

/// Create array of action/JointViaMotion messages.
/**
 * It allocates the memory for the array and calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__create(size_t size);

/// Destroy array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * array);

/// Check for action/JointViaMotion message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * rhs);

/// Copy an array of action/JointViaMotion messages.
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
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * output);

/// Initialize action/JointViaMotion message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request
 * )) before or use
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__init(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * msg);

/// Finalize action/JointViaMotion message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__fini(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * msg);

/// Create action/JointViaMotion message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request *
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__create();

/// Destroy action/JointViaMotion message.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__destroy(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * msg);

/// Check for action/JointViaMotion message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * lhs, const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * rhs);

/// Copy a action/JointViaMotion message.
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
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * input,
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * output);

/// Initialize array of action/JointViaMotion messages.
/**
 * It allocates the memory for the number of elements and calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * array, size_t size);

/// Finalize array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * array);

/// Create array of action/JointViaMotion messages.
/**
 * It allocates the memory for the array and calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__create(size_t size);

/// Destroy array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * array);

/// Check for action/JointViaMotion message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * rhs);

/// Copy an array of action/JointViaMotion messages.
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
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * output);

/// Initialize action/JointViaMotion message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response
 * )) before or use
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__init(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * msg);

/// Finalize action/JointViaMotion message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__fini(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * msg);

/// Create action/JointViaMotion message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response *
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__create();

/// Destroy action/JointViaMotion message.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__destroy(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * msg);

/// Check for action/JointViaMotion message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * lhs, const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * rhs);

/// Copy a action/JointViaMotion message.
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
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * input,
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * output);

/// Initialize array of action/JointViaMotion messages.
/**
 * It allocates the memory for the number of elements and calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * array, size_t size);

/// Finalize array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * array);

/// Create array of action/JointViaMotion messages.
/**
 * It allocates the memory for the array and calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__create(size_t size);

/// Destroy array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * array);

/// Check for action/JointViaMotion message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * rhs);

/// Copy an array of action/JointViaMotion messages.
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
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * output);

/// Initialize action/JointViaMotion message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage
 * )) before or use
 * panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__init(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * msg);

/// Finalize action/JointViaMotion message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * msg);

/// Create action/JointViaMotion message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage *
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__create();

/// Destroy action/JointViaMotion message.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__destroy(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * msg);

/// Check for action/JointViaMotion message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * lhs, const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * rhs);

/// Copy a action/JointViaMotion message.
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
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * input,
  panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * output);

/// Initialize array of action/JointViaMotion messages.
/**
 * It allocates the memory for the number of elements and calls
 * panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * array, size_t size);

/// Finalize array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * array);

/// Create array of action/JointViaMotion messages.
/**
 * It allocates the memory for the array and calls
 * panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__create(size_t size);

/// Destroy array of action/JointViaMotion messages.
/**
 * It calls
 * panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
void
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * array);

/// Check for action/JointViaMotion message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_panda_motion_generator_msgs
bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * rhs);

/// Copy an array of action/JointViaMotion messages.
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
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__JOINT_VIA_MOTION__FUNCTIONS_H_
