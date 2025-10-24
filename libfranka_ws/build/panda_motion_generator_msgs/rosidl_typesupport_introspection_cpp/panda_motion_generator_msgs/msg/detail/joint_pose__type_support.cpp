// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from panda_motion_generator_msgs:msg/JointPose.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "panda_motion_generator_msgs/msg/detail/joint_pose__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace panda_motion_generator_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void JointPose_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) panda_motion_generator_msgs::msg::JointPose(_init);
}

void JointPose_fini_function(void * message_memory)
{
  auto typed_message = static_cast<panda_motion_generator_msgs::msg::JointPose *>(message_memory);
  typed_message->~JointPose();
}

size_t size_function__JointPose__q(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__JointPose__q(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__JointPose__q(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__JointPose__q(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__JointPose__q(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__JointPose__q(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__JointPose__q(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__JointPose__q(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember JointPose_message_member_array[1] = {
  {
    "q",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(panda_motion_generator_msgs::msg::JointPose, q),  // bytes offset in struct
    nullptr,  // default value
    size_function__JointPose__q,  // size() function pointer
    get_const_function__JointPose__q,  // get_const(index) function pointer
    get_function__JointPose__q,  // get(index) function pointer
    fetch_function__JointPose__q,  // fetch(index, &value) function pointer
    assign_function__JointPose__q,  // assign(index, value) function pointer
    resize_function__JointPose__q  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers JointPose_message_members = {
  "panda_motion_generator_msgs::msg",  // message namespace
  "JointPose",  // message name
  1,  // number of fields
  sizeof(panda_motion_generator_msgs::msg::JointPose),
  JointPose_message_member_array,  // message members
  JointPose_init_function,  // function to initialize message memory (memory has to be allocated)
  JointPose_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t JointPose_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &JointPose_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace panda_motion_generator_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<panda_motion_generator_msgs::msg::JointPose>()
{
  return &::panda_motion_generator_msgs::msg::rosidl_typesupport_introspection_cpp::JointPose_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, panda_motion_generator_msgs, msg, JointPose)() {
  return &::panda_motion_generator_msgs::msg::rosidl_typesupport_introspection_cpp::JointPose_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
