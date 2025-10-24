// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from mujoco_ros_msgs:msg/StateUint.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "mujoco_ros_msgs/msg/detail/state_uint__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace mujoco_ros_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void StateUint_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) mujoco_ros_msgs::msg::StateUint(_init);
}

void StateUint_fini_function(void * message_memory)
{
  auto typed_message = static_cast<mujoco_ros_msgs::msg::StateUint *>(message_memory);
  typed_message->~StateUint();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember StateUint_message_member_array[2] = {
  {
    "value",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs::msg::StateUint, value),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "description",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs::msg::StateUint, description),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers StateUint_message_members = {
  "mujoco_ros_msgs::msg",  // message namespace
  "StateUint",  // message name
  2,  // number of fields
  sizeof(mujoco_ros_msgs::msg::StateUint),
  StateUint_message_member_array,  // message members
  StateUint_init_function,  // function to initialize message memory (memory has to be allocated)
  StateUint_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t StateUint_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &StateUint_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace mujoco_ros_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<mujoco_ros_msgs::msg::StateUint>()
{
  return &::mujoco_ros_msgs::msg::rosidl_typesupport_introspection_cpp::StateUint_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, mujoco_ros_msgs, msg, StateUint)() {
  return &::mujoco_ros_msgs::msg::rosidl_typesupport_introspection_cpp::StateUint_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
