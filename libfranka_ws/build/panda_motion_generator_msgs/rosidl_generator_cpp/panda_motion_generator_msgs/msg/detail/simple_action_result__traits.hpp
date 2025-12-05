// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from panda_motion_generator_msgs:msg/SimpleActionResult.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__TRAITS_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "panda_motion_generator_msgs/msg/detail/simple_action_result__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace panda_motion_generator_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const SimpleActionResult & msg,
  std::ostream & out)
{
  out << "{";
  // member: state
  {
    out << "state: ";
    rosidl_generator_traits::value_to_yaml(msg.state, out);
    out << ", ";
  }

  // member: error
  {
    out << "error: ";
    rosidl_generator_traits::value_to_yaml(msg.error, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SimpleActionResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "state: ";
    rosidl_generator_traits::value_to_yaml(msg.state, out);
    out << "\n";
  }

  // member: error
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "error: ";
    rosidl_generator_traits::value_to_yaml(msg.error, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SimpleActionResult & msg, bool use_flow_style = false)
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
  const panda_motion_generator_msgs::msg::SimpleActionResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  panda_motion_generator_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use panda_motion_generator_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const panda_motion_generator_msgs::msg::SimpleActionResult & msg)
{
  return panda_motion_generator_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<panda_motion_generator_msgs::msg::SimpleActionResult>()
{
  return "panda_motion_generator_msgs::msg::SimpleActionResult";
}

template<>
inline const char * name<panda_motion_generator_msgs::msg::SimpleActionResult>()
{
  return "panda_motion_generator_msgs/msg/SimpleActionResult";
}

template<>
struct has_fixed_size<panda_motion_generator_msgs::msg::SimpleActionResult>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<panda_motion_generator_msgs::msg::SimpleActionResult>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<panda_motion_generator_msgs::msg::SimpleActionResult>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__TRAITS_HPP_
