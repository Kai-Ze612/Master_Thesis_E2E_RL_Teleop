// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from franka_msgs:msg/FrankaModel.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "franka_msgs/msg/detail/franka_model__rosidl_typesupport_introspection_c.h"
#include "franka_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "franka_msgs/msg/detail/franka_model__functions.h"
#include "franka_msgs/msg/detail/franka_model__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  franka_msgs__msg__FrankaModel__init(message_memory);
}

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_fini_function(void * message_memory)
{
  franka_msgs__msg__FrankaModel__fini(message_memory);
}

size_t franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__size_function__FrankaModel__coriolis(
  const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__coriolis(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__coriolis(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__fetch_function__FrankaModel__coriolis(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__coriolis(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__assign_function__FrankaModel__coriolis(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__coriolis(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__size_function__FrankaModel__ee_zero_jacobian(
  const void * untyped_member)
{
  (void)untyped_member;
  return 42;
}

const void * franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__ee_zero_jacobian(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__ee_zero_jacobian(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__fetch_function__FrankaModel__ee_zero_jacobian(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__ee_zero_jacobian(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__assign_function__FrankaModel__ee_zero_jacobian(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__ee_zero_jacobian(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__size_function__FrankaModel__ee_body_jacobian(
  const void * untyped_member)
{
  (void)untyped_member;
  return 42;
}

const void * franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__ee_body_jacobian(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__ee_body_jacobian(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__fetch_function__FrankaModel__ee_body_jacobian(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__ee_body_jacobian(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__assign_function__FrankaModel__ee_body_jacobian(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__ee_body_jacobian(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__size_function__FrankaModel__mass(
  const void * untyped_member)
{
  (void)untyped_member;
  return 49;
}

const void * franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__mass(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__mass(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__fetch_function__FrankaModel__mass(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__mass(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__assign_function__FrankaModel__mass(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__mass(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_message_member_array[5] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(franka_msgs__msg__FrankaModel, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "coriolis",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(franka_msgs__msg__FrankaModel, coriolis),  // bytes offset in struct
    NULL,  // default value
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__size_function__FrankaModel__coriolis,  // size() function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__coriolis,  // get_const(index) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__coriolis,  // get(index) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__fetch_function__FrankaModel__coriolis,  // fetch(index, &value) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__assign_function__FrankaModel__coriolis,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "ee_zero_jacobian",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    42,  // array size
    false,  // is upper bound
    offsetof(franka_msgs__msg__FrankaModel, ee_zero_jacobian),  // bytes offset in struct
    NULL,  // default value
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__size_function__FrankaModel__ee_zero_jacobian,  // size() function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__ee_zero_jacobian,  // get_const(index) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__ee_zero_jacobian,  // get(index) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__fetch_function__FrankaModel__ee_zero_jacobian,  // fetch(index, &value) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__assign_function__FrankaModel__ee_zero_jacobian,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "ee_body_jacobian",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    42,  // array size
    false,  // is upper bound
    offsetof(franka_msgs__msg__FrankaModel, ee_body_jacobian),  // bytes offset in struct
    NULL,  // default value
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__size_function__FrankaModel__ee_body_jacobian,  // size() function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__ee_body_jacobian,  // get_const(index) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__ee_body_jacobian,  // get(index) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__fetch_function__FrankaModel__ee_body_jacobian,  // fetch(index, &value) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__assign_function__FrankaModel__ee_body_jacobian,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mass",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    49,  // array size
    false,  // is upper bound
    offsetof(franka_msgs__msg__FrankaModel, mass),  // bytes offset in struct
    NULL,  // default value
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__size_function__FrankaModel__mass,  // size() function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_const_function__FrankaModel__mass,  // get_const(index) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__get_function__FrankaModel__mass,  // get(index) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__fetch_function__FrankaModel__mass,  // fetch(index, &value) function pointer
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__assign_function__FrankaModel__mass,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_message_members = {
  "franka_msgs__msg",  // message namespace
  "FrankaModel",  // message name
  5,  // number of fields
  sizeof(franka_msgs__msg__FrankaModel),
  franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_message_member_array,  // message members
  franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_init_function,  // function to initialize message memory (memory has to be allocated)
  franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_message_type_support_handle = {
  0,
  &franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_franka_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, franka_msgs, msg, FrankaModel)() {
  franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_message_type_support_handle.typesupport_identifier) {
    franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &franka_msgs__msg__FrankaModel__rosidl_typesupport_introspection_c__FrankaModel_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
