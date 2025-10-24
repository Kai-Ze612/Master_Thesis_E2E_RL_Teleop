// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multi_mode_control_msgs:msg/JointGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__JOINT_GOAL__TRAITS_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__JOINT_GOAL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "multi_mode_control_msgs/msg/detail/joint_goal__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace multi_mode_control_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const JointGoal & msg,
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
    out << ", ";
  }

  // member: qd
  {
    if (msg.qd.size() == 0) {
      out << "qd: []";
    } else {
      out << "qd: [";
      size_t pending_items = msg.qd.size();
      for (auto item : msg.qd) {
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
  const JointGoal & msg,
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

  // member: qd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.qd.size() == 0) {
      out << "qd: []\n";
    } else {
      out << "qd:\n";
      for (auto item : msg.qd) {
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

inline std::string to_yaml(const JointGoal & msg, bool use_flow_style = false)
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

}  // namespace multi_mode_control_msgs

namespace rosidl_generator_traits
{

[[deprecated("use multi_mode_control_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const multi_mode_control_msgs::msg::JointGoal & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::msg::JointGoal & msg)
{
  return multi_mode_control_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::msg::JointGoal>()
{
  return "multi_mode_control_msgs::msg::JointGoal";
}

template<>
inline const char * name<multi_mode_control_msgs::msg::JointGoal>()
{
  return "multi_mode_control_msgs/msg/JointGoal";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::msg::JointGoal>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::msg::JointGoal>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<multi_mode_control_msgs::msg::JointGoal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__JOINT_GOAL__TRAITS_HPP_
