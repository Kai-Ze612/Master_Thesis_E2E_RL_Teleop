// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from mujoco_ros_msgs:srv/RegisterSensorNoiseModels.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__REGISTER_SENSOR_NOISE_MODELS__FUNCTIONS_H_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__REGISTER_SENSOR_NOISE_MODELS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "mujoco_ros_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "mujoco_ros_msgs/srv/detail/register_sensor_noise_models__struct.h"

/// Initialize srv/RegisterSensorNoiseModels message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request
 * )) before or use
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__init(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request * msg);

/// Finalize srv/RegisterSensorNoiseModels message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__fini(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request * msg);

/// Create srv/RegisterSensorNoiseModels message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request *
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__create();

/// Destroy srv/RegisterSensorNoiseModels message.
/**
 * It calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__destroy(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request * msg);

/// Check for srv/RegisterSensorNoiseModels message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__are_equal(const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request * lhs, const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request * rhs);

/// Copy a srv/RegisterSensorNoiseModels message.
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
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__copy(
  const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request * input,
  mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request * output);

/// Initialize array of srv/RegisterSensorNoiseModels messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence__init(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence * array, size_t size);

/// Finalize array of srv/RegisterSensorNoiseModels messages.
/**
 * It calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence__fini(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence * array);

/// Create array of srv/RegisterSensorNoiseModels messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence *
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence__create(size_t size);

/// Destroy array of srv/RegisterSensorNoiseModels messages.
/**
 * It calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence__destroy(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence * array);

/// Check for srv/RegisterSensorNoiseModels message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence__are_equal(const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence * lhs, const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence * rhs);

/// Copy an array of srv/RegisterSensorNoiseModels messages.
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
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence__copy(
  const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence * input,
  mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request__Sequence * output);

/// Initialize srv/RegisterSensorNoiseModels message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response
 * )) before or use
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__init(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response * msg);

/// Finalize srv/RegisterSensorNoiseModels message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__fini(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response * msg);

/// Create srv/RegisterSensorNoiseModels message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response *
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__create();

/// Destroy srv/RegisterSensorNoiseModels message.
/**
 * It calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__destroy(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response * msg);

/// Check for srv/RegisterSensorNoiseModels message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__are_equal(const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response * lhs, const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response * rhs);

/// Copy a srv/RegisterSensorNoiseModels message.
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
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__copy(
  const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response * input,
  mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response * output);

/// Initialize array of srv/RegisterSensorNoiseModels messages.
/**
 * It allocates the memory for the number of elements and calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence__init(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence * array, size_t size);

/// Finalize array of srv/RegisterSensorNoiseModels messages.
/**
 * It calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence__fini(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence * array);

/// Create array of srv/RegisterSensorNoiseModels messages.
/**
 * It allocates the memory for the array and calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence *
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence__create(size_t size);

/// Destroy array of srv/RegisterSensorNoiseModels messages.
/**
 * It calls
 * mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
void
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence__destroy(mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence * array);

/// Check for srv/RegisterSensorNoiseModels message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mujoco_ros_msgs
bool
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence__are_equal(const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence * lhs, const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence * rhs);

/// Copy an array of srv/RegisterSensorNoiseModels messages.
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
mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence__copy(
  const mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence * input,
  mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__REGISTER_SENSOR_NOISE_MODELS__FUNCTIONS_H_
