// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from franka_msgs:msg/PoseStampedArray.idl
// generated code does not contain a copyright notice

#ifndef FRANKA_MSGS__MSG__DETAIL__POSE_STAMPED_ARRAY__TRAITS_HPP_
#define FRANKA_MSGS__MSG__DETAIL__POSE_STAMPED_ARRAY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "franka_msgs/msg/detail/pose_stamped_array__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stampedposes'
#include "geometry_msgs/msg/detail/pose_stamped__traits.hpp"

namespace franka_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PoseStampedArray & msg,
  std::ostream & out)
{
  out << "{";
  // member: stampedposes
  {
    if (msg.stampedposes.size() == 0) {
      out << "stampedposes: []";
    } else {
      out << "stampedposes: [";
      size_t pending_items = msg.stampedposes.size();
      for (auto item : msg.stampedposes) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PoseStampedArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stampedposes
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.stampedposes.size() == 0) {
      out << "stampedposes: []\n";
    } else {
      out << "stampedposes:\n";
      for (auto item : msg.stampedposes) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PoseStampedArray & msg, bool use_flow_style = false)
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

}  // namespace franka_msgs

namespace rosidl_generator_traits
{

[[deprecated("use franka_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const franka_msgs::msg::PoseStampedArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  franka_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use franka_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const franka_msgs::msg::PoseStampedArray & msg)
{
  return franka_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<franka_msgs::msg::PoseStampedArray>()
{
  return "franka_msgs::msg::PoseStampedArray";
}

template<>
inline const char * name<franka_msgs::msg::PoseStampedArray>()
{
  return "franka_msgs/msg/PoseStampedArray";
}

template<>
struct has_fixed_size<franka_msgs::msg::PoseStampedArray>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<franka_msgs::msg::PoseStampedArray>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<franka_msgs::msg::PoseStampedArray>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // FRANKA_MSGS__MSG__DETAIL__POSE_STAMPED_ARRAY__TRAITS_HPP_
