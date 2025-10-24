// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:msg/GeomType.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_TYPE__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_TYPE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/msg/detail/geom_type__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace mujoco_ros_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const GeomType & msg,
  std::ostream & out)
{
  out << "{";
  // member: value
  {
    out << "value: ";
    rosidl_generator_traits::value_to_yaml(msg.value, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GeomType & msg,
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
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GeomType & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::msg::GeomType & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::msg::GeomType & msg)
{
  return mujoco_ros_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::msg::GeomType>()
{
  return "mujoco_ros_msgs::msg::GeomType";
}

template<>
inline const char * name<mujoco_ros_msgs::msg::GeomType>()
{
  return "mujoco_ros_msgs/msg/GeomType";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::msg::GeomType>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::msg::GeomType>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<mujoco_ros_msgs::msg::GeomType>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_TYPE__TRAITS_HPP_
