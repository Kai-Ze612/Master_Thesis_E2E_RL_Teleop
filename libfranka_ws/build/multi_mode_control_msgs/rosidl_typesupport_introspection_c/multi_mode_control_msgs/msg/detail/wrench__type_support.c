// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from multi_mode_control_msgs:msg/Wrench.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "multi_mode_control_msgs/msg/detail/wrench__rosidl_typesupport_introspection_c.h"
#include "multi_mode_control_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "multi_mode_control_msgs/msg/detail/wrench__functions.h"
#include "multi_mode_control_msgs/msg/detail/wrench__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multi_mode_control_msgs__msg__Wrench__init(message_memory);
}

void multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_fini_function(void * message_memory)
{
  multi_mode_control_msgs__msg__Wrench__fini(message_memory);
}

size_t multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__size_function__Wrench__data(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__get_const_function__Wrench__data(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__get_function__Wrench__data(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__fetch_function__Wrench__data(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__get_const_function__Wrench__data(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__assign_function__Wrench__data(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__get_function__Wrench__data(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_message_member_array[1] = {
  {
    "data",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__Wrench, data),  // bytes offset in struct
    NULL,  // default value
    multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__size_function__Wrench__data,  // size() function pointer
    multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__get_const_function__Wrench__data,  // get_const(index) function pointer
    multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__get_function__Wrench__data,  // get(index) function pointer
    multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__fetch_function__Wrench__data,  // fetch(index, &value) function pointer
    multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__assign_function__Wrench__data,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_message_members = {
  "multi_mode_control_msgs__msg",  // message namespace
  "Wrench",  // message name
  1,  // number of fields
  sizeof(multi_mode_control_msgs__msg__Wrench),
  multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_message_member_array,  // message members
  multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_init_function,  // function to initialize message memory (memory has to be allocated)
  multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_message_type_support_handle = {
  0,
  &multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multi_mode_control_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, msg, Wrench)() {
  if (!multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_message_type_support_handle.typesupport_identifier) {
    multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &multi_mode_control_msgs__msg__Wrench__rosidl_typesupport_introspection_c__Wrench_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
