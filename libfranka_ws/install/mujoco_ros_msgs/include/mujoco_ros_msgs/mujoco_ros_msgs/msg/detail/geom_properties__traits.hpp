// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:msg/GeomProperties.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/msg/detail/geom_properties__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'type'
#include "mujoco_ros_msgs/msg/detail/geom_type__traits.hpp"

namespace mujoco_ros_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const GeomProperties & msg,
  std::ostream & out)
{
  out << "{";
  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << ", ";
  }

  // member: type
  {
    out << "type: ";
    to_flow_style_yaml(msg.type, out);
    out << ", ";
  }

  // member: body_mass
  {
    out << "body_mass: ";
    rosidl_generator_traits::value_to_yaml(msg.body_mass, out);
    out << ", ";
  }

  // member: size_0
  {
    out << "size_0: ";
    rosidl_generator_traits::value_to_yaml(msg.size_0, out);
    out << ", ";
  }

  // member: size_1
  {
    out << "size_1: ";
    rosidl_generator_traits::value_to_yaml(msg.size_1, out);
    out << ", ";
  }

  // member: size_2
  {
    out << "size_2: ";
    rosidl_generator_traits::value_to_yaml(msg.size_2, out);
    out << ", ";
  }

  // member: friction_slide
  {
    out << "friction_slide: ";
    rosidl_generator_traits::value_to_yaml(msg.friction_slide, out);
    out << ", ";
  }

  // member: friction_spin
  {
    out << "friction_spin: ";
    rosidl_generator_traits::value_to_yaml(msg.friction_spin, out);
    out << ", ";
  }

  // member: friction_roll
  {
    out << "friction_roll: ";
    rosidl_generator_traits::value_to_yaml(msg.friction_roll, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GeomProperties & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }

  // member: type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "type:\n";
    to_block_style_yaml(msg.type, out, indentation + 2);
  }

  // member: body_mass
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "body_mass: ";
    rosidl_generator_traits::value_to_yaml(msg.body_mass, out);
    out << "\n";
  }

  // member: size_0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "size_0: ";
    rosidl_generator_traits::value_to_yaml(msg.size_0, out);
    out << "\n";
  }

  // member: size_1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "size_1: ";
    rosidl_generator_traits::value_to_yaml(msg.size_1, out);
    out << "\n";
  }

  // member: size_2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "size_2: ";
    rosidl_generator_traits::value_to_yaml(msg.size_2, out);
    out << "\n";
  }

  // member: friction_slide
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "friction_slide: ";
    rosidl_generator_traits::value_to_yaml(msg.friction_slide, out);
    out << "\n";
  }

  // member: friction_spin
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "friction_spin: ";
    rosidl_generator_traits::value_to_yaml(msg.friction_spin, out);
    out << "\n";
  }

  // member: friction_roll
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "friction_roll: ";
    rosidl_generator_traits::value_to_yaml(msg.friction_roll, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GeomProperties & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::msg::GeomProperties & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::msg::GeomProperties & msg)
{
  return mujoco_ros_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::msg::GeomProperties>()
{
  return "mujoco_ros_msgs::msg::GeomProperties";
}

template<>
inline const char * name<mujoco_ros_msgs::msg::GeomProperties>()
{
  return "mujoco_ros_msgs/msg/GeomProperties";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::msg::GeomProperties>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::msg::GeomProperties>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::msg::GeomProperties>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__TRAITS_HPP_
