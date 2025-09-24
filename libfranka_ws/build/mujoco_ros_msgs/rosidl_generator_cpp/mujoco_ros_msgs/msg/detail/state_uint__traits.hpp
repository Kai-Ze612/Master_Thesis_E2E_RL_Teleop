// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:msg/StateUint.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__STATE_UINT__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__STATE_UINT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/msg/detail/state_uint__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace mujoco_ros_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const StateUint & msg,
  std::ostream & out)
{
  out << "{";
  // member: value
  {
    out << "value: ";
    rosidl_generator_traits::value_to_yaml(msg.value, out);
    out << ", ";
  }

  // member: description
  {
    out << "description: ";
    rosidl_generator_traits::value_to_yaml(msg.description, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const StateUint & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: value
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "value: ";
    rosidl_generator_traits::value_to_yaml(msg.value, out);
    out << "\n";
  }

  // member: description
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "description: ";
    rosidl_generator_traits::value_to_yaml(msg.description, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const StateUint & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace mujoco_ros_msgs

namespace rosidl_generator_traits
{

[[deprecated("use mujoco_ros_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const mujoco_ros_msgs::msg::StateUint & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::msg::StateUint & msg)
{
  return mujoco_ros_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::msg::StateUint>()
{
  return "mujoco_ros_msgs::msg::StateUint";
}

template<>
inline const char * name<mujoco_ros_msgs::msg::StateUint>()
{
  return "mujoco_ros_msgs/msg/StateUint";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::msg::StateUint>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::msg::StateUint>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::msg::StateUint>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__STATE_UINT__TRAITS_HPP_
