// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mujoco_ros_msgs:srv/SetGravity.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mujoco_ros_msgs/srv/detail/set_gravity__rosidl_typesupport_introspection_c.h"
#include "mujoco_ros_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mujoco_ros_msgs/srv/detail/set_gravity__functions.h"
#include "mujoco_ros_msgs/srv/detail/set_gravity__struct.h"


// Include directives for member types
// Member `admin_hash`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mujoco_ros_msgs__srv__SetGravity_Request__init(message_memory);
}

void mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_fini_function(void * message_memory)
{
  mujoco_ros_msgs__srv__SetGravity_Request__fini(message_memory);
}

size_t mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__size_function__SetGravity_Request__gravity(
  const void * untyped_member)
{
  (void)untyped_member;
  return 3;
}

const void * mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__get_const_function__SetGravity_Request__gravity(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__get_function__SetGravity_Request__gravity(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__fetch_function__SetGravity_Request__gravity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__get_const_function__SetGravity_Request__gravity(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__assign_function__SetGravity_Request__gravity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__get_function__SetGravity_Request__gravity(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_message_member_array[2] = {
  {
    "admin_hash",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__srv__SetGravity_Request, admin_hash),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "gravity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    3,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__srv__SetGravity_Request, gravity),  // bytes offset in struct
    NULL,  // default value
    mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__size_function__SetGravity_Request__gravity,  // size() function pointer
    mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__get_const_function__SetGravity_Request__gravity,  // get_const(index) function pointer
    mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__get_function__SetGravity_Request__gravity,  // get(index) function pointer
    mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__fetch_function__SetGravity_Request__gravity,  // fetch(index, &value) function pointer
    mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__assign_function__SetGravity_Request__gravity,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_message_members = {
  "mujoco_ros_msgs__srv",  // message namespace
  "SetGravity_Request",  // message name
  2,  // number of fields
  sizeof(mujoco_ros_msgs__srv__SetGravity_Request),
  mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_message_member_array,  // message members
  mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_message_type_support_handle = {
  0,
  &mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mujoco_ros_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, SetGravity_Request)() {
  if (!mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_message_type_support_handle.typesupport_identifier) {
    mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mujoco_ros_msgs__srv__SetGravity_Request__rosidl_typesupport_introspection_c__SetGravity_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "mujoco_ros_msgs/srv/detail/set_gravity__rosidl_typesupport_introspection_c.h"
// already included above
// #include "mujoco_ros_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/set_gravity__functions.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/set_gravity__struct.h"


// Include directives for member types
// Member `status_message`
// already included above
// #include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mujoco_ros_msgs__srv__SetGravity_Response__init(message_memory);
}

void mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_fini_function(void * message_memory)
{
  mujoco_ros_msgs__srv__SetGravity_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_message_member_array[2] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__srv__SetGravity_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "status_message",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__srv__SetGravity_Response, status_message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_message_members = {
  "mujoco_ros_msgs__srv",  // message namespace
  "SetGravity_Response",  // message name
  2,  // number of fields
  sizeof(mujoco_ros_msgs__srv__SetGravity_Response),
  mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_message_member_array,  // message members
  mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_message_type_support_handle = {
  0,
  &mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mujoco_ros_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, SetGravity_Response)() {
  if (!mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_message_type_support_handle.typesupport_identifier) {
    mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mujoco_ros_msgs__srv__SetGravity_Response__rosidl_typesupport_introspection_c__SetGravity_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "mujoco_ros_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/set_gravity__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers mujoco_ros_msgs__srv__detail__set_gravity__rosidl_typesupport_introspection_c__SetGravity_service_members = {
  "mujoco_ros_msgs__srv",  // service namespace
  "SetGravity",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // mujoco_ros_msgs__srv__detail__set_gravity__rosidl_typesupport_introspection_c__SetGravity_Request_message_type_support_handle,
  NULL  // response message
  // mujoco_ros_msgs__srv__detail__set_gravity__rosidl_typesupport_introspection_c__SetGravity_Response_message_type_support_handle
};

static rosidl_service_type_support_t mujoco_ros_msgs__srv__detail__set_gravity__rosidl_typesupport_introspection_c__SetGravity_service_type_support_handle = {
  0,
  &mujoco_ros_msgs__srv__detail__set_gravity__rosidl_typesupport_introspection_c__SetGravity_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, SetGravity_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, SetGravity_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mujoco_ros_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, SetGravity)() {
  if (!mujoco_ros_msgs__srv__detail__set_gravity__rosidl_typesupport_introspection_c__SetGravity_service_type_support_handle.typesupport_identifier) {
    mujoco_ros_msgs__srv__detail__set_gravity__rosidl_typesupport_introspection_c__SetGravity_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)mujoco_ros_msgs__srv__detail__set_gravity__rosidl_typesupport_introspection_c__SetGravity_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, SetGravity_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, SetGravity_Response)()->data;
  }

  return &mujoco_ros_msgs__srv__detail__set_gravity__rosidl_typesupport_introspection_c__SetGravity_service_type_support_handle;
}
