// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from franka_msgs:msg/FrankaModel.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "franka_msgs/msg/detail/franka_model__struct.hpp"
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

void FrankaModel_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) franka_msgs::msg::FrankaModel(_init);
}

void FrankaModel_fini_function(void * message_memory)
{
  auto typed_message = static_cast<franka_msgs::msg::FrankaModel *>(message_memory);
  typed_message->~FrankaModel();
}

size_t size_function__FrankaModel__coriolis(const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * get_const_function__FrankaModel__coriolis(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void * get_function__FrankaModel__coriolis(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void fetch_function__FrankaModel__coriolis(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__FrankaModel__coriolis(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__FrankaModel__coriolis(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__FrankaModel__coriolis(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__FrankaModel__ee_zero_jacobian(const void * untyped_member)
{
  (void)untyped_member;
  return 42;
}

const void * get_const_function__FrankaModel__ee_zero_jacobian(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 42> *>(untyped_member);
  return &member[index];
}

void * get_function__FrankaModel__ee_zero_jacobian(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 42> *>(untyped_member);
  return &member[index];
}

void fetch_function__FrankaModel__ee_zero_jacobian(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__FrankaModel__ee_zero_jacobian(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__FrankaModel__ee_zero_jacobian(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__FrankaModel__ee_zero_jacobian(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__FrankaModel__ee_body_jacobian(const void * untyped_member)
{
  (void)untyped_member;
  return 42;
}

const void * get_const_function__FrankaModel__ee_body_jacobian(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 42> *>(untyped_member);
  return &member[index];
}

void * get_function__FrankaModel__ee_body_jacobian(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 42> *>(untyped_member);
  return &member[index];
}

void fetch_function__FrankaModel__ee_body_jacobian(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__FrankaModel__ee_body_jacobian(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__FrankaModel__ee_body_jacobian(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__FrankaModel__ee_body_jacobian(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__FrankaModel__mass(const void * untyped_member)
{
  (void)untyped_member;
  return 49;
}

const void * get_const_function__FrankaModel__mass(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 49> *>(untyped_member);
  return &member[index];
}

void * get_function__FrankaModel__mass(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 49> *>(untyped_member);
  return &member[index];
}

void fetch_function__FrankaModel__mass(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__FrankaModel__mass(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__FrankaModel__mass(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__FrankaModel__mass(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember FrankaModel_message_member_array[5] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(franka_msgs::msg::FrankaModel, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "coriolis",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(franka_msgs::msg::FrankaModel, coriolis),  // bytes offset in struct
    nullptr,  // default value
    size_function__FrankaModel__coriolis,  // size() function pointer
    get_const_function__FrankaModel__coriolis,  // get_const(index) function pointer
    get_function__FrankaModel__coriolis,  // get(index) function pointer
    fetch_function__FrankaModel__coriolis,  // fetch(index, &value) function pointer
    assign_function__FrankaModel__coriolis,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "ee_zero_jacobian",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    42,  // array size
    false,  // is upper bound
    offsetof(franka_msgs::msg::FrankaModel, ee_zero_jacobian),  // bytes offset in struct
    nullptr,  // default value
    size_function__FrankaModel__ee_zero_jacobian,  // size() function pointer
    get_const_function__FrankaModel__ee_zero_jacobian,  // get_const(index) function pointer
    get_function__FrankaModel__ee_zero_jacobian,  // get(index) function pointer
    fetch_function__FrankaModel__ee_zero_jacobian,  // fetch(index, &value) function pointer
    assign_function__FrankaModel__ee_zero_jacobian,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "ee_body_jacobian",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    42,  // array size
    false,  // is upper bound
    offsetof(franka_msgs::msg::FrankaModel, ee_body_jacobian),  // bytes offset in struct
    nullptr,  // default value
    size_function__FrankaModel__ee_body_jacobian,  // size() function pointer
    get_const_function__FrankaModel__ee_body_jacobian,  // get_const(index) function pointer
    get_function__FrankaModel__ee_body_jacobian,  // get(index) function pointer
    fetch_function__FrankaModel__ee_body_jacobian,  // fetch(index, &value) function pointer
    assign_function__FrankaModel__ee_body_jacobian,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "mass",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    49,  // array size
    false,  // is upper bound
    offsetof(franka_msgs::msg::FrankaModel, mass),  // bytes offset in struct
    nullptr,  // default value
    size_function__FrankaModel__mass,  // size() function pointer
    get_const_function__FrankaModel__mass,  // get_const(index) function pointer
    get_function__FrankaModel__mass,  // get(index) function pointer
    fetch_function__FrankaModel__mass,  // fetch(index, &value) function pointer
    assign_function__FrankaModel__mass,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers FrankaModel_message_members = {
  "franka_msgs::msg",  // message namespace
  "FrankaModel",  // message name
  5,  // number of fields
  sizeof(franka_msgs::msg::FrankaModel),
  FrankaModel_message_member_array,  // message members
  FrankaModel_init_function,  // function to initialize message memory (memory has to be allocated)
  FrankaModel_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t FrankaModel_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &FrankaModel_message_members,
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
get_message_type_support_handle<franka_msgs::msg::FrankaModel>()
{
  return &::franka_msgs::msg::rosidl_typesupport_introspection_cpp::FrankaModel_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, franka_msgs, msg, FrankaModel)() {
  return &::franka_msgs::msg::rosidl_typesupport_introspection_cpp::FrankaModel_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
