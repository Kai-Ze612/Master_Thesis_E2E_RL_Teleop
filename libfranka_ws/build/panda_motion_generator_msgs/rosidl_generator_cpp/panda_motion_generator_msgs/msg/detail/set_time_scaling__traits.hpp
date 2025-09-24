// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from panda_motion_generator_msgs:msg/SetTimeScaling.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__TRAITS_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "panda_motion_generator_msgs/msg/detail/set_time_scaling__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace panda_motion_generator_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const SetTimeScaling & msg,
  std::ostream & out)
{
  out << "{";
  // member: steps
  {
    out << "steps: ";
    rosidl_generator_traits::value_to_yaml(msg.steps, out);
    out << ", ";
  }

  // member: scaling
  {
    out << "scaling: ";
    rosidl_generator_traits::value_to_yaml(msg.scaling, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetTimeScaling & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: steps
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "steps: ";
    rosidl_generator_traits::value_to_yaml(msg.steps, out);
    out << "\n";
  }

  // member: scaling
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "scaling: ";
    rosidl_generator_traits::value_to_yaml(msg.scaling, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetTimeScaling & msg, bool use_flow_style = false)
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
  const panda_motion_generator_msgs::msg::SetTimeScaling & msg,
  std::ostream & out, size_t indentation = 0)
{
  panda_motion_generator_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use panda_motion_generator_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const panda_motion_generator_msgs::msg::SetTimeScaling & msg)
{
  return panda_motion_generator_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<panda_motion_generator_msgs::msg::SetTimeScaling>()
{
  return "panda_motion_generator_msgs::msg::SetTimeScaling";
}

template<>
inline const char * name<panda_motion_generator_msgs::msg::SetTimeScaling>()
{
  return "panda_motion_generator_msgs/msg/SetTimeScaling";
}

template<>
struct has_fixed_size<panda_motion_generator_msgs::msg::SetTimeScaling>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<panda_motion_generator_msgs::msg::SetTimeScaling>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<panda_motion_generator_msgs::msg::SetTimeScaling>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__TRAITS_HPP_
