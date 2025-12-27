// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from multi_mode_control_msgs:msg/Controller.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "multi_mode_control_msgs/msg/detail/controller__rosidl_typesupport_introspection_c.h"
#include "multi_mode_control_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "multi_mode_control_msgs/msg/detail/controller__functions.h"
#include "multi_mode_control_msgs/msg/detail/controller__struct.h"


// Include directives for member types
// Member `name`
// Member `resources`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multi_mode_control_msgs__msg__Controller__init(message_memory);
}

void multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_fini_function(void * message_memory)
{
  multi_mode_control_msgs__msg__Controller__fini(message_memory);
}

size_t multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__size_function__Controller__resources(
  const void * untyped_member)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return member->size;
}

const void * multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__get_const_function__Controller__resources(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void * multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__get_function__Controller__resources(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__fetch_function__Controller__resources(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const rosidl_runtime_c__String * item =
    ((const rosidl_runtime_c__String *)
    multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__get_const_function__Controller__resources(untyped_member, index));
  rosidl_runtime_c__String * value =
    (rosidl_runtime_c__String *)(untyped_value);
  *value = *item;
}

void multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__assign_function__Controller__resources(
  void * untyped_member, size_t index, const void * untyped_value)
{
  rosidl_runtime_c__String * item =
    ((rosidl_runtime_c__String *)
    multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__get_function__Controller__resources(untyped_member, index));
  const rosidl_runtime_c__String * value =
    (const rosidl_runtime_c__String *)(untyped_value);
  *item = *value;
}

bool multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__resize_function__Controller__resources(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  rosidl_runtime_c__String__Sequence__fini(member);
  return rosidl_runtime_c__String__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_message_member_array[2] = {
  {
    "name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__Controller, name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "resources",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__Controller, resources),  // bytes offset in struct
    NULL,  // default value
    multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__size_function__Controller__resources,  // size() function pointer
    multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__get_const_function__Controller__resources,  // get_const(index) function pointer
    multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__get_function__Controller__resources,  // get(index) function pointer
    multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__fetch_function__Controller__resources,  // fetch(index, &value) function pointer
    multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__assign_function__Controller__resources,  // assign(index, value) function pointer
    multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__resize_function__Controller__resources  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_message_members = {
  "multi_mode_control_msgs__msg",  // message namespace
  "Controller",  // message name
  2,  // number of fields
  sizeof(multi_mode_control_msgs__msg__Controller),
  multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_message_member_array,  // message members
  multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_init_function,  // function to initialize message memory (memory has to be allocated)
  multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_message_type_support_handle = {
  0,
  &multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multi_mode_control_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, msg, Controller)() {
  if (!multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_message_type_support_handle.typesupport_identifier) {
    multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &multi_mode_control_msgs__msg__Controller__rosidl_typesupport_introspection_c__Controller_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
