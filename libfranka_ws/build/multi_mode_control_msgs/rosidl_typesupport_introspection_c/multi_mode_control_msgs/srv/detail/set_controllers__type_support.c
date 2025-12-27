// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from multi_mode_control_msgs:srv/SetControllers.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "multi_mode_control_msgs/srv/detail/set_controllers__rosidl_typesupport_introspection_c.h"
#include "multi_mode_control_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "multi_mode_control_msgs/srv/detail/set_controllers__functions.h"
#include "multi_mode_control_msgs/srv/detail/set_controllers__struct.h"


// Include directives for member types
// Member `controllers`
#include "multi_mode_control_msgs/msg/controller.h"
// Member `controllers`
#include "multi_mode_control_msgs/msg/detail/controller__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multi_mode_control_msgs__srv__SetControllers_Request__init(message_memory);
}

void multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_fini_function(void * message_memory)
{
  multi_mode_control_msgs__srv__SetControllers_Request__fini(message_memory);
}

size_t multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__size_function__SetControllers_Request__controllers(
  const void * untyped_member)
{
  const multi_mode_control_msgs__msg__Controller__Sequence * member =
    (const multi_mode_control_msgs__msg__Controller__Sequence *)(untyped_member);
  return member->size;
}

const void * multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__get_const_function__SetControllers_Request__controllers(
  const void * untyped_member, size_t index)
{
  const multi_mode_control_msgs__msg__Controller__Sequence * member =
    (const multi_mode_control_msgs__msg__Controller__Sequence *)(untyped_member);
  return &member->data[index];
}

void * multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__get_function__SetControllers_Request__controllers(
  void * untyped_member, size_t index)
{
  multi_mode_control_msgs__msg__Controller__Sequence * member =
    (multi_mode_control_msgs__msg__Controller__Sequence *)(untyped_member);
  return &member->data[index];
}

void multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__fetch_function__SetControllers_Request__controllers(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const multi_mode_control_msgs__msg__Controller * item =
    ((const multi_mode_control_msgs__msg__Controller *)
    multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__get_const_function__SetControllers_Request__controllers(untyped_member, index));
  multi_mode_control_msgs__msg__Controller * value =
    (multi_mode_control_msgs__msg__Controller *)(untyped_value);
  *value = *item;
}

void multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__assign_function__SetControllers_Request__controllers(
  void * untyped_member, size_t index, const void * untyped_value)
{
  multi_mode_control_msgs__msg__Controller * item =
    ((multi_mode_control_msgs__msg__Controller *)
    multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__get_function__SetControllers_Request__controllers(untyped_member, index));
  const multi_mode_control_msgs__msg__Controller * value =
    (const multi_mode_control_msgs__msg__Controller *)(untyped_value);
  *item = *value;
}

bool multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__resize_function__SetControllers_Request__controllers(
  void * untyped_member, size_t size)
{
  multi_mode_control_msgs__msg__Controller__Sequence * member =
    (multi_mode_control_msgs__msg__Controller__Sequence *)(untyped_member);
  multi_mode_control_msgs__msg__Controller__Sequence__fini(member);
  return multi_mode_control_msgs__msg__Controller__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_message_member_array[1] = {
  {
    "controllers",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__srv__SetControllers_Request, controllers),  // bytes offset in struct
    NULL,  // default value
    multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__size_function__SetControllers_Request__controllers,  // size() function pointer
    multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__get_const_function__SetControllers_Request__controllers,  // get_const(index) function pointer
    multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__get_function__SetControllers_Request__controllers,  // get(index) function pointer
    multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__fetch_function__SetControllers_Request__controllers,  // fetch(index, &value) function pointer
    multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__assign_function__SetControllers_Request__controllers,  // assign(index, value) function pointer
    multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__resize_function__SetControllers_Request__controllers  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_message_members = {
  "multi_mode_control_msgs__srv",  // message namespace
  "SetControllers_Request",  // message name
  1,  // number of fields
  sizeof(multi_mode_control_msgs__srv__SetControllers_Request),
  multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_message_member_array,  // message members
  multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_message_type_support_handle = {
  0,
  &multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multi_mode_control_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetControllers_Request)() {
  multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, msg, Controller)();
  if (!multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_message_type_support_handle.typesupport_identifier) {
    multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &multi_mode_control_msgs__srv__SetControllers_Request__rosidl_typesupport_introspection_c__SetControllers_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "multi_mode_control_msgs/srv/detail/set_controllers__rosidl_typesupport_introspection_c.h"
// already included above
// #include "multi_mode_control_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "multi_mode_control_msgs/srv/detail/set_controllers__functions.h"
// already included above
// #include "multi_mode_control_msgs/srv/detail/set_controllers__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multi_mode_control_msgs__srv__SetControllers_Response__init(message_memory);
}

void multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_fini_function(void * message_memory)
{
  multi_mode_control_msgs__srv__SetControllers_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs__srv__SetControllers_Response, structure_needs_at_least_one_member),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_message_members = {
  "multi_mode_control_msgs__srv",  // message namespace
  "SetControllers_Response",  // message name
  1,  // number of fields
  sizeof(multi_mode_control_msgs__srv__SetControllers_Response),
  multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_message_member_array,  // message members
  multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_message_type_support_handle = {
  0,
  &multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multi_mode_control_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetControllers_Response)() {
  if (!multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_message_type_support_handle.typesupport_identifier) {
    multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &multi_mode_control_msgs__srv__SetControllers_Response__rosidl_typesupport_introspection_c__SetControllers_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "multi_mode_control_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "multi_mode_control_msgs/srv/detail/set_controllers__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers multi_mode_control_msgs__srv__detail__set_controllers__rosidl_typesupport_introspection_c__SetControllers_service_members = {
  "multi_mode_control_msgs__srv",  // service namespace
  "SetControllers",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // multi_mode_control_msgs__srv__detail__set_controllers__rosidl_typesupport_introspection_c__SetControllers_Request_message_type_support_handle,
  NULL  // response message
  // multi_mode_control_msgs__srv__detail__set_controllers__rosidl_typesupport_introspection_c__SetControllers_Response_message_type_support_handle
};

static rosidl_service_type_support_t multi_mode_control_msgs__srv__detail__set_controllers__rosidl_typesupport_introspection_c__SetControllers_service_type_support_handle = {
  0,
  &multi_mode_control_msgs__srv__detail__set_controllers__rosidl_typesupport_introspection_c__SetControllers_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetControllers_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetControllers_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multi_mode_control_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetControllers)() {
  if (!multi_mode_control_msgs__srv__detail__set_controllers__rosidl_typesupport_introspection_c__SetControllers_service_type_support_handle.typesupport_identifier) {
    multi_mode_control_msgs__srv__detail__set_controllers__rosidl_typesupport_introspection_c__SetControllers_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)multi_mode_control_msgs__srv__detail__set_controllers__rosidl_typesupport_introspection_c__SetControllers_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetControllers_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multi_mode_control_msgs, srv, SetControllers_Response)()->data;
  }

  return &multi_mode_control_msgs__srv__detail__set_controllers__rosidl_typesupport_introspection_c__SetControllers_service_type_support_handle;
}
