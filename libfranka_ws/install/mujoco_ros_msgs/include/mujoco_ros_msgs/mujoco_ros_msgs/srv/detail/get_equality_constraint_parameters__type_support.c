// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mujoco_ros_msgs:srv/GetEqualityConstraintParameters.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mujoco_ros_msgs/srv/detail/get_equality_constraint_parameters__rosidl_typesupport_introspection_c.h"
#include "mujoco_ros_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mujoco_ros_msgs/srv/detail/get_equality_constraint_parameters__functions.h"
#include "mujoco_ros_msgs/srv/detail/get_equality_constraint_parameters__struct.h"


// Include directives for member types
// Member `names`
// Member `admin_hash`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__init(message_memory);
}

void mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_fini_function(void * message_memory)
{
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__fini(message_memory);
}

size_t mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__size_function__GetEqualityConstraintParameters_Request__names(
  const void * untyped_member)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return member->size;
}

const void * mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__get_const_function__GetEqualityConstraintParameters_Request__names(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__get_function__GetEqualityConstraintParameters_Request__names(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__fetch_function__GetEqualityConstraintParameters_Request__names(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const rosidl_runtime_c__String * item =
    ((const rosidl_runtime_c__String *)
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__get_const_function__GetEqualityConstraintParameters_Request__names(untyped_member, index));
  rosidl_runtime_c__String * value =
    (rosidl_runtime_c__String *)(untyped_value);
  *value = *item;
}

void mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__assign_function__GetEqualityConstraintParameters_Request__names(
  void * untyped_member, size_t index, const void * untyped_value)
{
  rosidl_runtime_c__String * item =
    ((rosidl_runtime_c__String *)
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__get_function__GetEqualityConstraintParameters_Request__names(untyped_member, index));
  const rosidl_runtime_c__String * value =
    (const rosidl_runtime_c__String *)(untyped_value);
  *item = *value;
}

bool mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__resize_function__GetEqualityConstraintParameters_Request__names(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  rosidl_runtime_c__String__Sequence__fini(member);
  return rosidl_runtime_c__String__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_message_member_array[2] = {
  {
    "names",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request, names),  // bytes offset in struct
    NULL,  // default value
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__size_function__GetEqualityConstraintParameters_Request__names,  // size() function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__get_const_function__GetEqualityConstraintParameters_Request__names,  // get_const(index) function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__get_function__GetEqualityConstraintParameters_Request__names,  // get(index) function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__fetch_function__GetEqualityConstraintParameters_Request__names,  // fetch(index, &value) function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__assign_function__GetEqualityConstraintParameters_Request__names,  // assign(index, value) function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__resize_function__GetEqualityConstraintParameters_Request__names  // resize(index) function pointer
  },
  {
    "admin_hash",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request, admin_hash),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_message_members = {
  "mujoco_ros_msgs__srv",  // message namespace
  "GetEqualityConstraintParameters_Request",  // message name
  2,  // number of fields
  sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request),
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_message_member_array,  // message members
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_message_type_support_handle = {
  0,
  &mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mujoco_ros_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetEqualityConstraintParameters_Request)() {
  if (!mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_message_type_support_handle.typesupport_identifier) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "mujoco_ros_msgs/srv/detail/get_equality_constraint_parameters__rosidl_typesupport_introspection_c.h"
// already included above
// #include "mujoco_ros_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/get_equality_constraint_parameters__functions.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/get_equality_constraint_parameters__struct.h"


// Include directives for member types
// Member `parameters`
#include "mujoco_ros_msgs/msg/equality_constraint_parameters.h"
// Member `parameters`
#include "mujoco_ros_msgs/msg/detail/equality_constraint_parameters__rosidl_typesupport_introspection_c.h"
// Member `status_message`
// already included above
// #include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__init(message_memory);
}

void mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_fini_function(void * message_memory)
{
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__fini(message_memory);
}

size_t mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__size_function__GetEqualityConstraintParameters_Response__parameters(
  const void * untyped_member)
{
  const mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * member =
    (const mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence *)(untyped_member);
  return member->size;
}

const void * mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__get_const_function__GetEqualityConstraintParameters_Response__parameters(
  const void * untyped_member, size_t index)
{
  const mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * member =
    (const mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__get_function__GetEqualityConstraintParameters_Response__parameters(
  void * untyped_member, size_t index)
{
  mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * member =
    (mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence *)(untyped_member);
  return &member->data[index];
}

void mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__fetch_function__GetEqualityConstraintParameters_Response__parameters(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const mujoco_ros_msgs__msg__EqualityConstraintParameters * item =
    ((const mujoco_ros_msgs__msg__EqualityConstraintParameters *)
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__get_const_function__GetEqualityConstraintParameters_Response__parameters(untyped_member, index));
  mujoco_ros_msgs__msg__EqualityConstraintParameters * value =
    (mujoco_ros_msgs__msg__EqualityConstraintParameters *)(untyped_value);
  *value = *item;
}

void mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__assign_function__GetEqualityConstraintParameters_Response__parameters(
  void * untyped_member, size_t index, const void * untyped_value)
{
  mujoco_ros_msgs__msg__EqualityConstraintParameters * item =
    ((mujoco_ros_msgs__msg__EqualityConstraintParameters *)
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__get_function__GetEqualityConstraintParameters_Response__parameters(untyped_member, index));
  const mujoco_ros_msgs__msg__EqualityConstraintParameters * value =
    (const mujoco_ros_msgs__msg__EqualityConstraintParameters *)(untyped_value);
  *item = *value;
}

bool mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__resize_function__GetEqualityConstraintParameters_Response__parameters(
  void * untyped_member, size_t size)
{
  mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * member =
    (mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence *)(untyped_member);
  mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__fini(member);
  return mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_member_array[3] = {
  {
    "parameters",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response, parameters),  // bytes offset in struct
    NULL,  // default value
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__size_function__GetEqualityConstraintParameters_Response__parameters,  // size() function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__get_const_function__GetEqualityConstraintParameters_Response__parameters,  // get_const(index) function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__get_function__GetEqualityConstraintParameters_Response__parameters,  // get(index) function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__fetch_function__GetEqualityConstraintParameters_Response__parameters,  // fetch(index, &value) function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__assign_function__GetEqualityConstraintParameters_Response__parameters,  // assign(index, value) function pointer
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__resize_function__GetEqualityConstraintParameters_Response__parameters  // resize(index) function pointer
  },
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response, success),  // bytes offset in struct
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
    offsetof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response, status_message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_members = {
  "mujoco_ros_msgs__srv",  // message namespace
  "GetEqualityConstraintParameters_Response",  // message name
  3,  // number of fields
  sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response),
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_member_array,  // message members
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_type_support_handle = {
  0,
  &mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mujoco_ros_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetEqualityConstraintParameters_Response)() {
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, msg, EqualityConstraintParameters)();
  if (!mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_type_support_handle.typesupport_identifier) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "mujoco_ros_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/get_equality_constraint_parameters__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers mujoco_ros_msgs__srv__detail__get_equality_constraint_parameters__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_service_members = {
  "mujoco_ros_msgs__srv",  // service namespace
  "GetEqualityConstraintParameters",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // mujoco_ros_msgs__srv__detail__get_equality_constraint_parameters__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Request_message_type_support_handle,
  NULL  // response message
  // mujoco_ros_msgs__srv__detail__get_equality_constraint_parameters__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_Response_message_type_support_handle
};

static rosidl_service_type_support_t mujoco_ros_msgs__srv__detail__get_equality_constraint_parameters__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_service_type_support_handle = {
  0,
  &mujoco_ros_msgs__srv__detail__get_equality_constraint_parameters__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetEqualityConstraintParameters_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetEqualityConstraintParameters_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mujoco_ros_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetEqualityConstraintParameters)() {
  if (!mujoco_ros_msgs__srv__detail__get_equality_constraint_parameters__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_service_type_support_handle.typesupport_identifier) {
    mujoco_ros_msgs__srv__detail__get_equality_constraint_parameters__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)mujoco_ros_msgs__srv__detail__get_equality_constraint_parameters__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetEqualityConstraintParameters_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetEqualityConstraintParameters_Response)()->data;
  }

  return &mujoco_ros_msgs__srv__detail__get_equality_constraint_parameters__rosidl_typesupport_introspection_c__GetEqualityConstraintParameters_service_type_support_handle;
}
