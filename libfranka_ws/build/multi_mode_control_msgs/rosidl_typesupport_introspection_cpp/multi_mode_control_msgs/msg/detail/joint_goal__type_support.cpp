// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from multi_mode_control_msgs:msg/JointGoal.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "multi_mode_control_msgs/msg/detail/joint_goal__struct.hpp"
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

void JointGoal_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) multi_mode_control_msgs::msg::JointGoal(_init);
}

void JointGoal_fini_function(void * message_memory)
{
  auto typed_message = static_cast<multi_mode_control_msgs::msg::JointGoal *>(message_memory);
  typed_message->~JointGoal();
}

size_t size_function__JointGoal__q(const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * get_const_function__JointGoal__q(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void * get_function__JointGoal__q(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void fetch_function__JointGoal__q(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__JointGoal__q(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__JointGoal__q(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__JointGoal__q(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__JointGoal__qd(const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * get_const_function__JointGoal__qd(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void * get_function__JointGoal__qd(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void fetch_function__JointGoal__qd(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__JointGoal__qd(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__JointGoal__qd(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__JointGoal__qd(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember JointGoal_message_member_array[2] = {
  {
    "q",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs::msg::JointGoal, q),  // bytes offset in struct
    nullptr,  // default value
    size_function__JointGoal__q,  // size() function pointer
    get_const_function__JointGoal__q,  // get_const(index) function pointer
    get_function__JointGoal__q,  // get(index) function pointer
    fetch_function__JointGoal__q,  // fetch(index, &value) function pointer
    assign_function__JointGoal__q,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "qd",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(multi_mode_control_msgs::msg::JointGoal, qd),  // bytes offset in struct
    nullptr,  // default value
    size_function__JointGoal__qd,  // size() function pointer
    get_const_function__JointGoal__qd,  // get_const(index) function pointer
    get_function__JointGoal__qd,  // get(index) function pointer
    fetch_function__JointGoal__qd,  // fetch(index, &value) function pointer
    assign_function__JointGoal__qd,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers JointGoal_message_members = {
  "multi_mode_control_msgs::msg",  // message namespace
  "JointGoal",  // message name
  2,  // number of fields
  sizeof(multi_mode_control_msgs::msg::JointGoal),
  JointGoal_message_member_array,  // message members
  JointGoal_init_function,  // function to initialize message memory (memory has to be allocated)
  JointGoal_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t JointGoal_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &JointGoal_message_members,
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
get_message_type_support_handle<multi_mode_control_msgs::msg::JointGoal>()
{
  return &::multi_mode_control_msgs::msg::rosidl_typesupport_introspection_cpp::JointGoal_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, multi_mode_control_msgs, msg, JointGoal)() {
  return &::multi_mode_control_msgs::msg::rosidl_typesupport_introspection_cpp::JointGoal_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
