// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from panda_motion_generator_msgs:msg/JointPose.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__TRAITS_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "panda_motion_generator_msgs/msg/detail/joint_pose__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace panda_motion_generator_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const JointPose & msg,
  std::ostream & out)
{
  out << "{";
  // member: q
  {
    if (msg.q.size() == 0) {
      out << "q: []";
    } else {
      out << "q: [";
      size_t pending_items = msg.q.size();
      for (auto item : msg.q) {
        rosidl_generator_traits::value_to_yaml(item, out);
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
  const JointPose & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: q
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.q.size() == 0) {
      out << "q: []\n";
    } else {
      out << "q:\n";
      for (auto item : msg.q) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const JointPose & msg, bool use_flow_style = false)
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

}  // namespace panda_motion_generator_msgs

namespace rosidl_generator_traits
{

[[deprecated("use panda_motion_generator_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const panda_motion_generator_msgs::msg::JointPose & msg,
  std::ostream & out, size_t indentation = 0)
{
  panda_motion_generator_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use panda_motion_generator_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const panda_motion_generator_msgs::msg::JointPose & msg)
{
  return panda_motion_generator_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<panda_motion_generator_msgs::msg::JointPose>()
{
  return "panda_motion_generator_msgs::msg::JointPose";
}

template<>
inline const char * name<panda_motion_generator_msgs::msg::JointPose>()
{
  return "panda_motion_generator_msgs/msg/JointPose";
}

template<>
struct has_fixed_size<panda_motion_generator_msgs::msg::JointPose>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<panda_motion_generator_msgs::msg::JointPose>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<panda_motion_generator_msgs::msg::JointPose>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__TRAITS_HPP_
