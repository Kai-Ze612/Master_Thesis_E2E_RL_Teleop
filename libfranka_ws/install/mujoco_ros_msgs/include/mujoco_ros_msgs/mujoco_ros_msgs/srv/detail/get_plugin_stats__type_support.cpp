// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from mujoco_ros_msgs:srv/GetPluginStats.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "mujoco_ros_msgs/srv/detail/get_plugin_stats__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace mujoco_ros_msgs
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void GetPluginStats_Request_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) mujoco_ros_msgs::srv::GetPluginStats_Request(_init);
}

void GetPluginStats_Request_fini_function(void * message_memory)
{
  auto typed_message = static_cast<mujoco_ros_msgs::srv::GetPluginStats_Request *>(message_memory);
  typed_message->~GetPluginStats_Request();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GetPluginStats_Request_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs::srv::GetPluginStats_Request, structure_needs_at_least_one_member),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GetPluginStats_Request_message_members = {
  "mujoco_ros_msgs::srv",  // message namespace
  "GetPluginStats_Request",  // message name
  1,  // number of fields
  sizeof(mujoco_ros_msgs::srv::GetPluginStats_Request),
  GetPluginStats_Request_message_member_array,  // message members
  GetPluginStats_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  GetPluginStats_Request_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GetPluginStats_Request_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GetPluginStats_Request_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace mujoco_ros_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<mujoco_ros_msgs::srv::GetPluginStats_Request>()
{
  return &::mujoco_ros_msgs::srv::rosidl_typesupport_introspection_cpp::GetPluginStats_Request_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, mujoco_ros_msgs, srv, GetPluginStats_Request)() {
  return &::mujoco_ros_msgs::srv::rosidl_typesupport_introspection_cpp::GetPluginStats_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "array"
// already included above
// #include "cstddef"
// already included above
// #include "string"
// already included above
// #include "vector"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/get_plugin_stats__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/field_types.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace mujoco_ros_msgs
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void GetPluginStats_Response_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) mujoco_ros_msgs::srv::GetPluginStats_Response(_init);
}

void GetPluginStats_Response_fini_function(void * message_memory)
{
  auto typed_message = static_cast<mujoco_ros_msgs::srv::GetPluginStats_Response *>(message_memory);
  typed_message->~GetPluginStats_Response();
}

size_t size_function__GetPluginStats_Response__stats(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<mujoco_ros_msgs::msg::PluginStats> *>(untyped_member);
  return member->size();
}

const void * get_const_function__GetPluginStats_Response__stats(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<mujoco_ros_msgs::msg::PluginStats> *>(untyped_member);
  return &member[index];
}

void * get_function__GetPluginStats_Response__stats(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<mujoco_ros_msgs::msg::PluginStats> *>(untyped_member);
  return &member[index];
}

void fetch_function__GetPluginStats_Response__stats(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const mujoco_ros_msgs::msg::PluginStats *>(
    get_const_function__GetPluginStats_Response__stats(untyped_member, index));
  auto & value = *reinterpret_cast<mujoco_ros_msgs::msg::PluginStats *>(untyped_value);
  value = item;
}

void assign_function__GetPluginStats_Response__stats(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<mujoco_ros_msgs::msg::PluginStats *>(
    get_function__GetPluginStats_Response__stats(untyped_member, index));
  const auto & value = *reinterpret_cast<const mujoco_ros_msgs::msg::PluginStats *>(untyped_value);
  item = value;
}

void resize_function__GetPluginStats_Response__stats(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<mujoco_ros_msgs::msg::PluginStats> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GetPluginStats_Response_message_member_array[1] = {
  {
    "stats",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<mujoco_ros_msgs::msg::PluginStats>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs::srv::GetPluginStats_Response, stats),  // bytes offset in struct
    nullptr,  // default value
    size_function__GetPluginStats_Response__stats,  // size() function pointer
    get_const_function__GetPluginStats_Response__stats,  // get_const(index) function pointer
    get_function__GetPluginStats_Response__stats,  // get(index) function pointer
    fetch_function__GetPluginStats_Response__stats,  // fetch(index, &value) function pointer
    assign_function__GetPluginStats_Response__stats,  // assign(index, value) function pointer
    resize_function__GetPluginStats_Response__stats  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GetPluginStats_Response_message_members = {
  "mujoco_ros_msgs::srv",  // message namespace
  "GetPluginStats_Response",  // message name
  1,  // number of fields
  sizeof(mujoco_ros_msgs::srv::GetPluginStats_Response),
  GetPluginStats_Response_message_member_array,  // message members
  GetPluginStats_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  GetPluginStats_Response_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GetPluginStats_Response_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GetPluginStats_Response_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace mujoco_ros_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<mujoco_ros_msgs::srv::GetPluginStats_Response>()
{
  return &::mujoco_ros_msgs::srv::rosidl_typesupport_introspection_cpp::GetPluginStats_Response_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, mujoco_ros_msgs, srv, GetPluginStats_Response)() {
  return &::mujoco_ros_msgs::srv::rosidl_typesupport_introspection_cpp::GetPluginStats_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/get_plugin_stats__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/service_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/service_type_support_decl.hpp"

namespace mujoco_ros_msgs
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

// this is intentionally not const to allow initialization later to prevent an initialization race
static ::rosidl_typesupport_introspection_cpp::ServiceMembers GetPluginStats_service_members = {
  "mujoco_ros_msgs::srv",  // service namespace
  "GetPluginStats",  // service name
  // these two fields are initialized below on the first access
  // see get_service_type_support_handle<mujoco_ros_msgs::srv::GetPluginStats>()
  nullptr,  // request message
  nullptr  // response message
};

static const rosidl_service_type_support_t GetPluginStats_service_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GetPluginStats_service_members,
  get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace mujoco_ros_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<mujoco_ros_msgs::srv::GetPluginStats>()
{
  // get a handle to the value to be returned
  auto service_type_support =
    &::mujoco_ros_msgs::srv::rosidl_typesupport_introspection_cpp::GetPluginStats_service_type_support_handle;
  // get a non-const and properly typed version of the data void *
  auto service_members = const_cast<::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
    static_cast<const ::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
      service_type_support->data));
  // make sure that both the request_members_ and the response_members_ are initialized
  // if they are not, initialize them
  if (
    service_members->request_members_ == nullptr ||
    service_members->response_members_ == nullptr)
  {
    // initialize the request_members_ with the static function from the external library
    service_members->request_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::mujoco_ros_msgs::srv::GetPluginStats_Request
      >()->data
      );
    // initialize the response_members_ with the static function from the external library
    service_members->response_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::mujoco_ros_msgs::srv::GetPluginStats_Response
      >()->data
      );
  }
  // finally return the properly initialized service_type_support handle
  return service_type_support;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, mujoco_ros_msgs, srv, GetPluginStats)() {
  return ::rosidl_typesupport_introspection_cpp::get_service_type_support_handle<mujoco_ros_msgs::srv::GetPluginStats>();
}

#ifdef __cplusplus
}
#endif
