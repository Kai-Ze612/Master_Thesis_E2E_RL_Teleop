// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from multi_mode_control_msgs:srv/SetCartesianImpedance.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "multi_mode_control_msgs/srv/detail/set_cartesian_impedance__rosidl_typesupport_introspection_c.h"
#include "multi_mode_control_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "multi_mode_control_msgs/srv/detail/set_cartesian_impedance__functions.h"
#include "multi_mode_control_msgs/srv/detail/set_cartesian_impedance__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multi_mode_control_msgs__srv__SetCartesianImpedance_Request__init(message_memory);
}

void multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_fini_function(void * message_memory)
{
  multi_mode_control_msgs__srv__SetCartesianImpedance_Request__fini(message_memory);
}

size_t multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__size_function__SetCartesianImpedance_Request__stiffness(
  const void * untyped_member)
{
  (void)untyped_member;
  return 36;
}

const void * multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_const_function__SetCartesianImpedance_Request__stiffness(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_function__SetCartesianImpedance_Request__stiffness(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__fetch_function__SetCartesianImpedance_Request__stiffness(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_const_function__SetCartesianImpedance_Request__stiffness(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__assign_function__SetCartesianImpedance_Request__stiffness(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_function__SetCartesianImpedance_Request__stiffness(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__size_function__SetCartesianImpedance_Request__damping_ratio(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_const_function__SetCartesianImpedance_Request__damping_ratio(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_function__SetCartesianImpedance_Request__damping_ratio(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__fetch_function__SetCartesianImpedance_Request__damping_ratio(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_const_function__SetCartesianImpedance_Request__damping_ratio(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__assign_function__SetCartesianImpedance_Request__damping_ratio(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_function__SetCartesianImpedance_Request__damping_ratio(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_message_member_array[3] = {
  {
    "stiffness",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    36,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__srv__SetCartesianImpedance_Request, stiffness),  // bytes offset in struct
    NULL,  // default value
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__size_function__SetCartesianImpedance_Request__stiffness,  // size() function pointer
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_const_function__SetCartesianImpedance_Request__stiffness,  // get_const(index) function pointer
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_function__SetCartesianImpedance_Request__stiffness,  // get(index) function pointer
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__fetch_function__SetCartesianImpedance_Request__stiffness,  // fetch(index, &value) function pointer
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__assign_function__SetCartesianImpedance_Request__stiffness,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "damping_ratio",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__srv__SetCartesianImpedance_Request, damping_ratio),  // bytes offset in struct
    NULL,  // default value
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__size_function__SetCartesianImpedance_Request__damping_ratio,  // size() function pointer
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_const_function__SetCartesianImpedance_Request__damping_ratio,  // get_const(index) function pointer
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__get_function__SetCartesianImpedance_Request__damping_ratio,  // get(index) function pointer
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__fetch_function__SetCartesianImpedance_Request__damping_ratio,  // fetch(index, &value) function pointer
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__assign_function__SetCartesianImpedance_Request__damping_ratio,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "nullspace_stiffness",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__srv__SetCartesianImpedance_Request, nullspace_stiffness),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_message_members = {
  "multi_mode_control_msgs__srv",  // message namespace
  "SetCartesianImpedance_Request",  // message name
  3,  // number of fields
  sizeof(multi_mode_control_msgs__srv__SetCartesianImpedance_Request),
  multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_message_member_array,  // message members
  multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_message_type_support_handle = {
  0,
  &multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multi_mode_control_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetCartesianImpedance_Request)() {
  if (!multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_message_type_support_handle.typesupport_identifier) {
    multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &multi_mode_control_msgs__srv__SetCartesianImpedance_Request__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "multi_mode_control_msgs/srv/detail/set_cartesian_impedance__rosidl_typesupport_introspection_c.h"
// already included above
// #include "multi_mode_control_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "multi_mode_control_msgs/srv/detail/set_cartesian_impedance__functions.h"
// already included above
// #include "multi_mode_control_msgs/srv/detail/set_cartesian_impedance__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multi_mode_control_msgs__srv__SetCartesianImpedance_Response__init(message_memory);
}

void multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_fini_function(void * message_memory)
{
  multi_mode_control_msgs__srv__SetCartesianImpedance_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__srv__SetCartesianImpedance_Response, structure_needs_at_least_one_member),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_message_members = {
  "multi_mode_control_msgs__srv",  // message namespace
  "SetCartesianImpedance_Response",  // message name
  1,  // number of fields
  sizeof(multi_mode_control_msgs__srv__SetCartesianImpedance_Response),
  multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_message_member_array,  // message members
  multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_message_type_support_handle = {
  0,
  &multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multi_mode_control_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetCartesianImpedance_Response)() {
  if (!multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_message_type_support_handle.typesupport_identifier) {
    multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &multi_mode_control_msgs__srv__SetCartesianImpedance_Response__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "multi_mode_control_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "multi_mode_control_msgs/srv/detail/set_cartesian_impedance__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers multi_mode_control_msgs__srv__detail__set_cartesian_impedance__rosidl_typesupport_introspection_c__SetCartesianImpedance_service_members = {
  "multi_mode_control_msgs__srv",  // service namespace
  "SetCartesianImpedance",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // multi_mode_control_msgs__srv__detail__set_cartesian_impedance__rosidl_typesupport_introspection_c__SetCartesianImpedance_Request_message_type_support_handle,
  NULL  // response message
  // multi_mode_control_msgs__srv__detail__set_cartesian_impedance__rosidl_typesupport_introspection_c__SetCartesianImpedance_Response_message_type_support_handle
};

static rosidl_service_type_support_t multi_mode_control_msgs__srv__detail__set_cartesian_impedance__rosidl_typesupport_introspection_c__SetCartesianImpedance_service_type_support_handle = {
  0,
  &multi_mode_control_msgs__srv__detail__set_cartesian_impedance__rosidl_typesupport_introspection_c__SetCartesianImpedance_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetCartesianImpedance_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetCartesianImpedance_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multi_mode_control_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetCartesianImpedance)() {
  if (!multi_mode_control_msgs__srv__detail__set_cartesian_impedance__rosidl_typesupport_introspection_c__SetCartesianImpedance_service_type_support_handle.typesupport_identifier) {
    multi_mode_control_msgs__srv__detail__set_cartesian_impedance__rosidl_typesupport_introspection_c__SetCartesianImpedance_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)multi_mode_control_msgs__srv__detail__set_cartesian_impedance__rosidl_typesupport_introspection_c__SetCartesianImpedance_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetCartesianImpedance_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetCartesianImpedance_Response)()->data;
  }

  return &multi_mode_control_msgs__srv__detail__set_cartesian_impedance__rosidl_typesupport_introspection_c__SetCartesianImpedance_service_type_support_handle;
}
