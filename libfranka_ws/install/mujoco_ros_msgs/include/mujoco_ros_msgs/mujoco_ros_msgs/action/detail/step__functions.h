// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from mujoco_ros_msgs:action/Step.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__ACTION__DETAIL__STEP__FUNCTIONS_H_
#define MUJOCO_ROS_MSGS__ACTION__DETAIL__STEP__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "mujoco_ros_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "mujoco_ros_msgs/action/detail/step__struct.h"

/// Initialize action/Step message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__action__Step_Goal
 * )) before or use
 * mujoco_ros_msgs__action__Step_Goal__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Goal__init(mujoco_ros_msgs__action__Step_Goal * msg);

/// Finalize action/Step message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Goal__fini(mujoco_ros_msgs__action__Step_Goal * msg);

/// Create action/Step message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__action__Step_Goal__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_Goal *
mujoco_ros_msgs__action__Step_Goal__create();

/// Destroy action/Step message.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_Goal__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Goal__destroy(mujoco_ros_msgs__action__Step_Goal * msg);

/// Check for action/Step message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Goal__are_equal(const mujoco_ros_msgs__action__Step_Goal * lhs, const mujoco_ros_msgs__action__Step_Goal * rhs);

/// Copy a action/Step message.
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
mujoco_ros_msgs__action__Step_Goal__copy(
  const mujoco_ros_msgs__action__Step_Goal * input,
  mujoco_ros_msgs__action__Step_Goal * output);

/// Initialize array of action/Step messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__action__Step_Goal__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Goal__Sequence__init(mujoco_ros_msgs__action__Step_Goal__Sequence * array, size_t size);

/// Finalize array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_Goal__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Goal__Sequence__fini(mujoco_ros_msgs__action__Step_Goal__Sequence * array);

/// Create array of action/Step messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__action__Step_Goal__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_Goal__Sequence *
mujoco_ros_msgs__action__Step_Goal__Sequence__create(size_t size);

/// Destroy array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_Goal__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Goal__Sequence__destroy(mujoco_ros_msgs__action__Step_Goal__Sequence * array);

/// Check for action/Step message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Goal__Sequence__are_equal(const mujoco_ros_msgs__action__Step_Goal__Sequence * lhs, const mujoco_ros_msgs__action__Step_Goal__Sequence * rhs);

/// Copy an array of action/Step messages.
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
mujoco_ros_msgs__action__Step_Goal__Sequence__copy(
  const mujoco_ros_msgs__action__Step_Goal__Sequence * input,
  mujoco_ros_msgs__action__Step_Goal__Sequence * output);

/// Initialize action/Step message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__action__Step_Result
 * )) before or use
 * mujoco_ros_msgs__action__Step_Result__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Result__init(mujoco_ros_msgs__action__Step_Result * msg);

/// Finalize action/Step message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Result__fini(mujoco_ros_msgs__action__Step_Result * msg);

/// Create action/Step message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__action__Step_Result__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_Result *
mujoco_ros_msgs__action__Step_Result__create();

/// Destroy action/Step message.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_Result__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Result__destroy(mujoco_ros_msgs__action__Step_Result * msg);

/// Check for action/Step message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Result__are_equal(const mujoco_ros_msgs__action__Step_Result * lhs, const mujoco_ros_msgs__action__Step_Result * rhs);

/// Copy a action/Step message.
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
mujoco_ros_msgs__action__Step_Result__copy(
  const mujoco_ros_msgs__action__Step_Result * input,
  mujoco_ros_msgs__action__Step_Result * output);

/// Initialize array of action/Step messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__action__Step_Result__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Result__Sequence__init(mujoco_ros_msgs__action__Step_Result__Sequence * array, size_t size);

/// Finalize array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_Result__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Result__Sequence__fini(mujoco_ros_msgs__action__Step_Result__Sequence * array);

/// Create array of action/Step messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__action__Step_Result__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_Result__Sequence *
mujoco_ros_msgs__action__Step_Result__Sequence__create(size_t size);

/// Destroy array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_Result__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Result__Sequence__destroy(mujoco_ros_msgs__action__Step_Result__Sequence * array);

/// Check for action/Step message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Result__Sequence__are_equal(const mujoco_ros_msgs__action__Step_Result__Sequence * lhs, const mujoco_ros_msgs__action__Step_Result__Sequence * rhs);

/// Copy an array of action/Step messages.
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
mujoco_ros_msgs__action__Step_Result__Sequence__copy(
  const mujoco_ros_msgs__action__Step_Result__Sequence * input,
  mujoco_ros_msgs__action__Step_Result__Sequence * output);

/// Initialize action/Step message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__action__Step_Feedback
 * )) before or use
 * mujoco_ros_msgs__action__Step_Feedback__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Feedback__init(mujoco_ros_msgs__action__Step_Feedback * msg);

/// Finalize action/Step message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Feedback__fini(mujoco_ros_msgs__action__Step_Feedback * msg);

/// Create action/Step message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__action__Step_Feedback__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_Feedback *
mujoco_ros_msgs__action__Step_Feedback__create();

/// Destroy action/Step message.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_Feedback__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Feedback__destroy(mujoco_ros_msgs__action__Step_Feedback * msg);

/// Check for action/Step message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Feedback__are_equal(const mujoco_ros_msgs__action__Step_Feedback * lhs, const mujoco_ros_msgs__action__Step_Feedback * rhs);

/// Copy a action/Step message.
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
mujoco_ros_msgs__action__Step_Feedback__copy(
  const mujoco_ros_msgs__action__Step_Feedback * input,
  mujoco_ros_msgs__action__Step_Feedback * output);

/// Initialize array of action/Step messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__action__Step_Feedback__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Feedback__Sequence__init(mujoco_ros_msgs__action__Step_Feedback__Sequence * array, size_t size);

/// Finalize array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_Feedback__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Feedback__Sequence__fini(mujoco_ros_msgs__action__Step_Feedback__Sequence * array);

/// Create array of action/Step messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__action__Step_Feedback__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_Feedback__Sequence *
mujoco_ros_msgs__action__Step_Feedback__Sequence__create(size_t size);

/// Destroy array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_Feedback__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_Feedback__Sequence__destroy(mujoco_ros_msgs__action__Step_Feedback__Sequence * array);

/// Check for action/Step message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_Feedback__Sequence__are_equal(const mujoco_ros_msgs__action__Step_Feedback__Sequence * lhs, const mujoco_ros_msgs__action__Step_Feedback__Sequence * rhs);

/// Copy an array of action/Step messages.
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
mujoco_ros_msgs__action__Step_Feedback__Sequence__copy(
  const mujoco_ros_msgs__action__Step_Feedback__Sequence * input,
  mujoco_ros_msgs__action__Step_Feedback__Sequence * output);

/// Initialize action/Step message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__action__Step_SendGoal_Request
 * )) before or use
 * mujoco_ros_msgs__action__Step_SendGoal_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_SendGoal_Request__init(mujoco_ros_msgs__action__Step_SendGoal_Request * msg);

/// Finalize action/Step message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_SendGoal_Request__fini(mujoco_ros_msgs__action__Step_SendGoal_Request * msg);

/// Create action/Step message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__action__Step_SendGoal_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_SendGoal_Request *
mujoco_ros_msgs__action__Step_SendGoal_Request__create();

/// Destroy action/Step message.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_SendGoal_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_SendGoal_Request__destroy(mujoco_ros_msgs__action__Step_SendGoal_Request * msg);

/// Check for action/Step message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_SendGoal_Request__are_equal(const mujoco_ros_msgs__action__Step_SendGoal_Request * lhs, const mujoco_ros_msgs__action__Step_SendGoal_Request * rhs);

/// Copy a action/Step message.
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
mujoco_ros_msgs__action__Step_SendGoal_Request__copy(
  const mujoco_ros_msgs__action__Step_SendGoal_Request * input,
  mujoco_ros_msgs__action__Step_SendGoal_Request * output);

/// Initialize array of action/Step messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__action__Step_SendGoal_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__init(mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * array, size_t size);

/// Finalize array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_SendGoal_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__fini(mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * array);

/// Create array of action/Step messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence *
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__create(size_t size);

/// Destroy array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__destroy(mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * array);

/// Check for action/Step message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__are_equal(const mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * lhs, const mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * rhs);

/// Copy an array of action/Step messages.
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
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__copy(
  const mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * input,
  mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * output);

/// Initialize action/Step message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__action__Step_SendGoal_Response
 * )) before or use
 * mujoco_ros_msgs__action__Step_SendGoal_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_SendGoal_Response__init(mujoco_ros_msgs__action__Step_SendGoal_Response * msg);

/// Finalize action/Step message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_SendGoal_Response__fini(mujoco_ros_msgs__action__Step_SendGoal_Response * msg);

/// Create action/Step message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__action__Step_SendGoal_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_SendGoal_Response *
mujoco_ros_msgs__action__Step_SendGoal_Response__create();

/// Destroy action/Step message.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_SendGoal_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_SendGoal_Response__destroy(mujoco_ros_msgs__action__Step_SendGoal_Response * msg);

/// Check for action/Step message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_SendGoal_Response__are_equal(const mujoco_ros_msgs__action__Step_SendGoal_Response * lhs, const mujoco_ros_msgs__action__Step_SendGoal_Response * rhs);

/// Copy a action/Step message.
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
mujoco_ros_msgs__action__Step_SendGoal_Response__copy(
  const mujoco_ros_msgs__action__Step_SendGoal_Response * input,
  mujoco_ros_msgs__action__Step_SendGoal_Response * output);

/// Initialize array of action/Step messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__action__Step_SendGoal_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__init(mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * array, size_t size);

/// Finalize array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_SendGoal_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__fini(mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * array);

/// Create array of action/Step messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence *
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__create(size_t size);

/// Destroy array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__destroy(mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * array);

/// Check for action/Step message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__are_equal(const mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * lhs, const mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * rhs);

/// Copy an array of action/Step messages.
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
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__copy(
  const mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * input,
  mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * output);

/// Initialize action/Step message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__action__Step_GetResult_Request
 * )) before or use
 * mujoco_ros_msgs__action__Step_GetResult_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_GetResult_Request__init(mujoco_ros_msgs__action__Step_GetResult_Request * msg);

/// Finalize action/Step message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_GetResult_Request__fini(mujoco_ros_msgs__action__Step_GetResult_Request * msg);

/// Create action/Step message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__action__Step_GetResult_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_GetResult_Request *
mujoco_ros_msgs__action__Step_GetResult_Request__create();

/// Destroy action/Step message.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_GetResult_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_GetResult_Request__destroy(mujoco_ros_msgs__action__Step_GetResult_Request * msg);

/// Check for action/Step message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_GetResult_Request__are_equal(const mujoco_ros_msgs__action__Step_GetResult_Request * lhs, const mujoco_ros_msgs__action__Step_GetResult_Request * rhs);

/// Copy a action/Step message.
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
mujoco_ros_msgs__action__Step_GetResult_Request__copy(
  const mujoco_ros_msgs__action__Step_GetResult_Request * input,
  mujoco_ros_msgs__action__Step_GetResult_Request * output);

/// Initialize array of action/Step messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__action__Step_GetResult_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__init(mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * array, size_t size);

/// Finalize array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_GetResult_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__fini(mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * array);

/// Create array of action/Step messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence *
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__create(size_t size);

/// Destroy array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__destroy(mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * array);

/// Check for action/Step message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__are_equal(const mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * lhs, const mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * rhs);

/// Copy an array of action/Step messages.
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
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__copy(
  const mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * input,
  mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * output);

/// Initialize action/Step message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__action__Step_GetResult_Response
 * )) before or use
 * mujoco_ros_msgs__action__Step_GetResult_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_GetResult_Response__init(mujoco_ros_msgs__action__Step_GetResult_Response * msg);

/// Finalize action/Step message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_GetResult_Response__fini(mujoco_ros_msgs__action__Step_GetResult_Response * msg);

/// Create action/Step message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__action__Step_GetResult_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_GetResult_Response *
mujoco_ros_msgs__action__Step_GetResult_Response__create();

/// Destroy action/Step message.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_GetResult_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_GetResult_Response__destroy(mujoco_ros_msgs__action__Step_GetResult_Response * msg);

/// Check for action/Step message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_GetResult_Response__are_equal(const mujoco_ros_msgs__action__Step_GetResult_Response * lhs, const mujoco_ros_msgs__action__Step_GetResult_Response * rhs);

/// Copy a action/Step message.
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
mujoco_ros_msgs__action__Step_GetResult_Response__copy(
  const mujoco_ros_msgs__action__Step_GetResult_Response * input,
  mujoco_ros_msgs__action__Step_GetResult_Response * output);

/// Initialize array of action/Step messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__action__Step_GetResult_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__init(mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * array, size_t size);

/// Finalize array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_GetResult_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__fini(mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * array);

/// Create array of action/Step messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence *
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__create(size_t size);

/// Destroy array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__destroy(mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * array);

/// Check for action/Step message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__are_equal(const mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * lhs, const mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * rhs);

/// Copy an array of action/Step messages.
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
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__copy(
  const mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * input,
  mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * output);

/// Initialize action/Step message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__action__Step_FeedbackMessage
 * )) before or use
 * mujoco_ros_msgs__action__Step_FeedbackMessage__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_FeedbackMessage__init(mujoco_ros_msgs__action__Step_FeedbackMessage * msg);

/// Finalize action/Step message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_FeedbackMessage__fini(mujoco_ros_msgs__action__Step_FeedbackMessage * msg);

/// Create action/Step message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__action__Step_FeedbackMessage__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_FeedbackMessage *
mujoco_ros_msgs__action__Step_FeedbackMessage__create();

/// Destroy action/Step message.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_FeedbackMessage__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_FeedbackMessage__destroy(mujoco_ros_msgs__action__Step_FeedbackMessage * msg);

/// Check for action/Step message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_FeedbackMessage__are_equal(const mujoco_ros_msgs__action__Step_FeedbackMessage * lhs, const mujoco_ros_msgs__action__Step_FeedbackMessage * rhs);

/// Copy a action/Step message.
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
mujoco_ros_msgs__action__Step_FeedbackMessage__copy(
  const mujoco_ros_msgs__action__Step_FeedbackMessage * input,
  mujoco_ros_msgs__action__Step_FeedbackMessage * output);

/// Initialize array of action/Step messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__action__Step_FeedbackMessage__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__init(mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * array, size_t size);

/// Finalize array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_FeedbackMessage__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__fini(mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * array);

/// Create array of action/Step messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence *
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__create(size_t size);

/// Destroy array of action/Step messages.
/**
 * It calls
 * mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__destroy(mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * array);

/// Check for action/Step message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__are_equal(const mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * lhs, const mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * rhs);

/// Copy an array of action/Step messages.
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
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__copy(
  const mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * input,
  mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__ACTION__DETAIL__STEP__FUNCTIONS_H_
