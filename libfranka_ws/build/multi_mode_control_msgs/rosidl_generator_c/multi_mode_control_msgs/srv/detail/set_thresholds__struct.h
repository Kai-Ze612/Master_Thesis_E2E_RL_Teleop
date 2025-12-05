// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from multi_mode_control_msgs:srv/SetThresholds.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_THRESHOLDS__STRUCT_H_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_THRESHOLDS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'level1'
// Member 'level2'
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__struct.h"

/// Struct defined in srv/SetThresholds in the package multi_mode_control_msgs.
typedef struct multi_mode_control_msgs__srv__SetThresholds_Request
{
  multi_mode_control_msgs__msg__ContactThresholds level1;
  multi_mode_control_msgs__msg__ContactThresholds level2;
} multi_mode_control_msgs__srv__SetThresholds_Request;

// Struct for a sequence of multi_mode_control_msgs__srv__SetThresholds_Request.
typedef struct multi_mode_control_msgs__srv__SetThresholds_Request__Sequence
{
  multi_mode_control_msgs__srv__SetThresholds_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} multi_mode_control_msgs__srv__SetThresholds_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetThresholds in the package multi_mode_control_msgs.
typedef struct multi_mode_control_msgs__srv__SetThresholds_Response
{
  uint8_t structure_needs_at_least_one_member;
} multi_mode_control_msgs__srv__SetThresholds_Response;

// Struct for a sequence of multi_mode_control_msgs__srv__SetThresholds_Response.
typedef struct multi_mode_control_msgs__srv__SetThresholds_Response__Sequence
{
  multi_mode_control_msgs__srv__SetThresholds_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} multi_mode_control_msgs__srv__SetThresholds_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_THRESHOLDS__STRUCT_H_
