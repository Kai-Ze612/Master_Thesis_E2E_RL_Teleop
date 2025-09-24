// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from multi_mode_control_msgs:srv/SetWorldState.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_WORLD_STATE__STRUCT_H_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_WORLD_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'HUMAN_PRESENT'.
enum
{
  multi_mode_control_msgs__srv__SetWorldState_Request__HUMAN_PRESENT = 0l
};

/// Constant 'OBJECT_IN_HAND'.
enum
{
  multi_mode_control_msgs__srv__SetWorldState_Request__OBJECT_IN_HAND = 1l
};

// Include directives for member types
// Member 'items'
// Member 'values'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/SetWorldState in the package multi_mode_control_msgs.
typedef struct multi_mode_control_msgs__srv__SetWorldState_Request
{
  rosidl_runtime_c__int32__Sequence items;
  rosidl_runtime_c__boolean__Sequence values;
} multi_mode_control_msgs__srv__SetWorldState_Request;

// Struct for a sequence of multi_mode_control_msgs__srv__SetWorldState_Request.
typedef struct multi_mode_control_msgs__srv__SetWorldState_Request__Sequence
{
  multi_mode_control_msgs__srv__SetWorldState_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} multi_mode_control_msgs__srv__SetWorldState_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetWorldState in the package multi_mode_control_msgs.
typedef struct multi_mode_control_msgs__srv__SetWorldState_Response
{
  uint8_t structure_needs_at_least_one_member;
} multi_mode_control_msgs__srv__SetWorldState_Response;

// Struct for a sequence of multi_mode_control_msgs__srv__SetWorldState_Response.
typedef struct multi_mode_control_msgs__srv__SetWorldState_Response__Sequence
{
  multi_mode_control_msgs__srv__SetWorldState_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} multi_mode_control_msgs__srv__SetWorldState_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_WORLD_STATE__STRUCT_H_
