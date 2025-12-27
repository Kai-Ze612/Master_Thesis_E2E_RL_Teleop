// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from multi_mode_control_msgs:msg/Wrench.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "multi_mode_control_msgs/msg/detail/wrench__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace multi_mode_control_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void Wrench_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) multi_mode_control_msgs::msg::Wrench(_init);
}

void Wrench_fini_function(void * message_memory)
{
  auto typed_message = static_cast<multi_mode_control_msgs::msg::Wrench *>(message_memory);
  typed_message->~Wrench();
}

size_t size_function__Wrench__data(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__Wrench__data(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__Wrench__data(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__Wrench__data(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__Wrench__data(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__Wrench__data(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__Wrench__data(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Wrench_message_member_array[1] = {
  {
    "data",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs::msg::Wrench, data),  // bytes offset in struct
    nullptr,  // default value
    size_function__Wrench__data,  // size() function pointer
    get_const_function__Wrench__data,  // get_const(index) function pointer
    get_function__Wrench__data,  // get(index) function pointer
    fetch_function__Wrench__data,  // fetch(index, &value) function pointer
    assign_function__Wrench__data,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Wrench_message_members = {
  "multi_mode_control_msgs::msg",  // message namespace
  "Wrench",  // message name
  1,  // number of fields
  sizeof(multi_mode_control_msgs::msg::Wrench),
  Wrench_message_member_array,  // message members
  Wrench_init_function,  // function to initialize message memory (memory has to be allocated)
  Wrench_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Wrench_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Wrench_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace multi_mode_control_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<multi_mode_control_msgs::msg::Wrench>()
{
  return &::multi_mode_control_msgs::msg::rosidl_typesupport_introspection_cpp::Wrench_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, multi_mode_control_msgs, msg, Wrench)() {
  return &::multi_mode_control_msgs::msg::rosidl_typesupport_introspection_cpp::Wrench_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
