// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from multi_mode_control_msgs:msg/ContactThresholds.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__rosidl_typesupport_introspection_c.h"
#include "multi_mode_control_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__functions.h"
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__struct.h"


// Include directives for member types
// Member `tau_max`
// Member `tau_min`
#include "multi_mode_control_msgs/msg/joint_array.h"
// Member `tau_max`
// Member `tau_min`
#include "multi_mode_control_msgs/msg/detail/joint_array__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multi_mode_control_msgs__msg__ContactThresholds__init(message_memory);
}

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_fini_function(void * message_memory)
{
  multi_mode_control_msgs__msg__ContactThresholds__fini(message_memory);
}

size_t multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__size_function__ContactThresholds__f_max(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__f_max(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__f_max(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__fetch_function__ContactThresholds__f_max(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__f_max(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__assign_function__ContactThresholds__f_max(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__f_max(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__size_function__ContactThresholds__f_min(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__f_min(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__f_min(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__fetch_function__ContactThresholds__f_min(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__f_min(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__assign_function__ContactThresholds__f_min(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__f_min(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__size_function__ContactThresholds__tau_max(
  const void * untyped_member)
{
  const multi_mode_control_msgs__msg__JointArray__Sequence * member =
    (const multi_mode_control_msgs__msg__JointArray__Sequence *)(untyped_member);
  return member->size;
}

const void * multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__tau_max(
  const void * untyped_member, size_t index)
{
  const multi_mode_control_msgs__msg__JointArray__Sequence * member =
    (const multi_mode_control_msgs__msg__JointArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__tau_max(
  void * untyped_member, size_t index)
{
  multi_mode_control_msgs__msg__JointArray__Sequence * member =
    (multi_mode_control_msgs__msg__JointArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__fetch_function__ContactThresholds__tau_max(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const multi_mode_control_msgs__msg__JointArray * item =
    ((const multi_mode_control_msgs__msg__JointArray *)
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__tau_max(untyped_member, index));
  multi_mode_control_msgs__msg__JointArray * value =
    (multi_mode_control_msgs__msg__JointArray *)(untyped_value);
  *value = *item;
}

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__assign_function__ContactThresholds__tau_max(
  void * untyped_member, size_t index, const void * untyped_value)
{
  multi_mode_control_msgs__msg__JointArray * item =
    ((multi_mode_control_msgs__msg__JointArray *)
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__tau_max(untyped_member, index));
  const multi_mode_control_msgs__msg__JointArray * value =
    (const multi_mode_control_msgs__msg__JointArray *)(untyped_value);
  *item = *value;
}

bool multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__resize_function__ContactThresholds__tau_max(
  void * untyped_member, size_t size)
{
  multi_mode_control_msgs__msg__JointArray__Sequence * member =
    (multi_mode_control_msgs__msg__JointArray__Sequence *)(untyped_member);
  multi_mode_control_msgs__msg__JointArray__Sequence__fini(member);
  return multi_mode_control_msgs__msg__JointArray__Sequence__init(member, size);
}

size_t multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__size_function__ContactThresholds__tau_min(
  const void * untyped_member)
{
  const multi_mode_control_msgs__msg__JointArray__Sequence * member =
    (const multi_mode_control_msgs__msg__JointArray__Sequence *)(untyped_member);
  return member->size;
}

const void * multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__tau_min(
  const void * untyped_member, size_t index)
{
  const multi_mode_control_msgs__msg__JointArray__Sequence * member =
    (const multi_mode_control_msgs__msg__JointArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__tau_min(
  void * untyped_member, size_t index)
{
  multi_mode_control_msgs__msg__JointArray__Sequence * member =
    (multi_mode_control_msgs__msg__JointArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__fetch_function__ContactThresholds__tau_min(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const multi_mode_control_msgs__msg__JointArray * item =
    ((const multi_mode_control_msgs__msg__JointArray *)
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__tau_min(untyped_member, index));
  multi_mode_control_msgs__msg__JointArray * value =
    (multi_mode_control_msgs__msg__JointArray *)(untyped_value);
  *value = *item;
}

void multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__assign_function__ContactThresholds__tau_min(
  void * untyped_member, size_t index, const void * untyped_value)
{
  multi_mode_control_msgs__msg__JointArray * item =
    ((multi_mode_control_msgs__msg__JointArray *)
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__tau_min(untyped_member, index));
  const multi_mode_control_msgs__msg__JointArray * value =
    (const multi_mode_control_msgs__msg__JointArray *)(untyped_value);
  *item = *value;
}

bool multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__resize_function__ContactThresholds__tau_min(
  void * untyped_member, size_t size)
{
  multi_mode_control_msgs__msg__JointArray__Sequence * member =
    (multi_mode_control_msgs__msg__JointArray__Sequence *)(untyped_member);
  multi_mode_control_msgs__msg__JointArray__Sequence__fini(member);
  return multi_mode_control_msgs__msg__JointArray__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_member_array[9] = {
  {
    "f_max",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__ContactThresholds, f_max),  // bytes offset in struct
    NULL,  // default value
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__size_function__ContactThresholds__f_max,  // size() function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__f_max,  // get_const(index) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__f_max,  // get(index) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__fetch_function__ContactThresholds__f_max,  // fetch(index, &value) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__assign_function__ContactThresholds__f_max,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_min",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__ContactThresholds, f_min),  // bytes offset in struct
    NULL,  // default value
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__size_function__ContactThresholds__f_min,  // size() function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__f_min,  // get_const(index) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__f_min,  // get(index) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__fetch_function__ContactThresholds__f_min,  // fetch(index, &value) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__assign_function__ContactThresholds__f_min,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "tau_max",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__ContactThresholds, tau_max),  // bytes offset in struct
    NULL,  // default value
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__size_function__ContactThresholds__tau_max,  // size() function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__tau_max,  // get_const(index) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__tau_max,  // get(index) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__fetch_function__ContactThresholds__tau_max,  // fetch(index, &value) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__assign_function__ContactThresholds__tau_max,  // assign(index, value) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__resize_function__ContactThresholds__tau_max  // resize(index) function pointer
  },
  {
    "tau_min",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__ContactThresholds, tau_min),  // bytes offset in struct
    NULL,  // default value
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__size_function__ContactThresholds__tau_min,  // size() function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_const_function__ContactThresholds__tau_min,  // get_const(index) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__get_function__ContactThresholds__tau_min,  // get(index) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__fetch_function__ContactThresholds__tau_min,  // fetch(index, &value) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__assign_function__ContactThresholds__tau_min,  // assign(index, value) function pointer
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__resize_function__ContactThresholds__tau_min  // resize(index) function pointer
  },
  {
    "f_abs_max",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__ContactThresholds, f_abs_max),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "m_abs_max",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__ContactThresholds, m_abs_max),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "abs_mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__ContactThresholds, abs_mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "directional_mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__ContactThresholds, directional_mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "torque_mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__msg__ContactThresholds, torque_mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_members = {
  "multi_mode_control_msgs__msg",  // message namespace
  "ContactThresholds",  // message name
  9,  // number of fields
  sizeof(multi_mode_control_msgs__msg__ContactThresholds),
  multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_member_array,  // message members
  multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_init_function,  // function to initialize message memory (memory has to be allocated)
  multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_type_support_handle = {
  0,
  &multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multi_mode_control_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, msg, ContactThresholds)() {
  multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, msg, JointArray)();
  multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, msg, JointArray)();
  if (!multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_type_support_handle.typesupport_identifier) {
    multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &multi_mode_control_msgs__msg__ContactThresholds__rosidl_typesupport_introspection_c__ContactThresholds_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
