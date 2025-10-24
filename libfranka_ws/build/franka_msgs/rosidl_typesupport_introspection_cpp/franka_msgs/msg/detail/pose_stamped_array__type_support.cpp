// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from franka_msgs:msg/PoseStampedArray.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "franka_msgs/msg/detail/pose_stamped_array__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace franka_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void PoseStampedArray_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) franka_msgs::msg::PoseStampedArray(_init);
}

void PoseStampedArray_fini_function(void * message_memory)
{
  auto typed_message = static_cast<franka_msgs::msg::PoseStampedArray *>(message_memory);
  typed_message->~PoseStampedArray();
}

size_t size_function__PoseStampedArray__stampedposes(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::PoseStamped> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PoseStampedArray__stampedposes(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::PoseStamped> *>(untyped_member);
  return &member[index];
}

void * get_function__PoseStampedArray__stampedposes(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::PoseStamped> *>(untyped_member);
  return &member[index];
}

void fetch_function__PoseStampedArray__stampedposes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const geometry_msgs::msg::PoseStamped *>(
    get_const_function__PoseStampedArray__stampedposes(untyped_member, index));
  auto & value = *reinterpret_cast<geometry_msgs::msg::PoseStamped *>(untyped_value);
  value = item;
}

void assign_function__PoseStampedArray__stampedposes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<geometry_msgs::msg::PoseStamped *>(
    get_function__PoseStampedArray__stampedposes(untyped_member, index));
  const auto & value = *reinterpret_cast<const geometry_msgs::msg::PoseStamped *>(untyped_value);
  item = value;
}

void resize_function__PoseStampedArray__stampedposes(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::PoseStamped> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PoseStampedArray_message_member_array[1] = {
  {
    "stampedposes",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::PoseStamped>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(franka_msgs::msg::PoseStampedArray, stampedposes),  // bytes offset in struct
    nullptr,  // default value
    size_function__PoseStampedArray__stampedposes,  // size() function pointer
    get_const_function__PoseStampedArray__stampedposes,  // get_const(index) function pointer
    get_function__PoseStampedArray__stampedposes,  // get(index) function pointer
    fetch_function__PoseStampedArray__stampedposes,  // fetch(index, &value) function pointer
    assign_function__PoseStampedArray__stampedposes,  // assign(index, value) function pointer
    resize_function__PoseStampedArray__stampedposes  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PoseStampedArray_message_members = {
  "franka_msgs::msg",  // message namespace
  "PoseStampedArray",  // message name
  1,  // number of fields
  sizeof(franka_msgs::msg::PoseStampedArray),
  PoseStampedArray_message_member_array,  // message members
  PoseStampedArray_init_function,  // function to initialize message memory (memory has to be allocated)
  PoseStampedArray_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PoseStampedArray_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PoseStampedArray_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace franka_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<franka_msgs::msg::PoseStampedArray>()
{
  return &::franka_msgs::msg::rosidl_typesupport_introspection_cpp::PoseStampedArray_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, franka_msgs, msg, PoseStampedArray)() {
  return &::franka_msgs::msg::rosidl_typesupport_introspection_cpp::PoseStampedArray_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
