// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from mujoco_ros_msgs:msg/SensorNoiseModel.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "mujoco_ros_msgs/msg/detail/sensor_noise_model__struct.hpp"
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

void SensorNoiseModel_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) mujoco_ros_msgs::msg::SensorNoiseModel(_init);
}

void SensorNoiseModel_fini_function(void * message_memory)
{
  auto typed_message = static_cast<mujoco_ros_msgs::msg::SensorNoiseModel *>(message_memory);
  typed_message->~SensorNoiseModel();
}

size_t size_function__SensorNoiseModel__mean(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SensorNoiseModel__mean(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__SensorNoiseModel__mean(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__SensorNoiseModel__mean(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__SensorNoiseModel__mean(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__SensorNoiseModel__mean(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__SensorNoiseModel__mean(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__SensorNoiseModel__mean(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__SensorNoiseModel__std(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SensorNoiseModel__std(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__SensorNoiseModel__std(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__SensorNoiseModel__std(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__SensorNoiseModel__std(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__SensorNoiseModel__std(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__SensorNoiseModel__std(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__SensorNoiseModel__std(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember SensorNoiseModel_message_member_array[4] = {
  {
    "sensor_name",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs::msg::SensorNoiseModel, sensor_name),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "mean",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs::msg::SensorNoiseModel, mean),  // bytes offset in struct
    nullptr,  // default value
    size_function__SensorNoiseModel__mean,  // size() function pointer
    get_const_function__SensorNoiseModel__mean,  // get_const(index) function pointer
    get_function__SensorNoiseModel__mean,  // get(index) function pointer
    fetch_function__SensorNoiseModel__mean,  // fetch(index, &value) function pointer
    assign_function__SensorNoiseModel__mean,  // assign(index, value) function pointer
    resize_function__SensorNoiseModel__mean  // resize(index) function pointer
  },
  {
    "std",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs::msg::SensorNoiseModel, std),  // bytes offset in struct
    nullptr,  // default value
    size_function__SensorNoiseModel__std,  // size() function pointer
    get_const_function__SensorNoiseModel__std,  // get_const(index) function pointer
    get_function__SensorNoiseModel__std,  // get(index) function pointer
    fetch_function__SensorNoiseModel__std,  // fetch(index, &value) function pointer
    assign_function__SensorNoiseModel__std,  // assign(index, value) function pointer
    resize_function__SensorNoiseModel__std  // resize(index) function pointer
  },
  {
    "set_flag",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs::msg::SensorNoiseModel, set_flag),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers SensorNoiseModel_message_members = {
  "mujoco_ros_msgs::msg",  // message namespace
  "SensorNoiseModel",  // message name
  4,  // number of fields
  sizeof(mujoco_ros_msgs::msg::SensorNoiseModel),
  SensorNoiseModel_message_member_array,  // message members
  SensorNoiseModel_init_function,  // function to initialize message memory (memory has to be allocated)
  SensorNoiseModel_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t SensorNoiseModel_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &SensorNoiseModel_message_members,
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
get_message_type_support_handle<mujoco_ros_msgs::msg::SensorNoiseModel>()
{
  return &::mujoco_ros_msgs::msg::rosidl_typesupport_introspection_cpp::SensorNoiseModel_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, mujoco_ros_msgs, msg, SensorNoiseModel)() {
  return &::mujoco_ros_msgs::msg::rosidl_typesupport_introspection_cpp::SensorNoiseModel_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
