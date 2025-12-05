// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multi_mode_control_msgs:msg/Controller.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__TRAITS_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "multi_mode_control_msgs/msg/detail/controller__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace multi_mode_control_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Controller & msg,
  std::ostream & out)
{
  out << "{";
  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << ", ";
  }

  // member: resources
  {
    if (msg.resources.size() == 0) {
      out << "resources: []";
    } else {
      out << "resources: [";
      size_t pending_items = msg.resources.size();
      for (auto item : msg.resources) {
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
  const Controller & msg,
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

  // member: resources
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.resources.size() == 0) {
      out << "resources: []\n";
    } else {
      out << "resources:\n";
      for (auto item : msg.resources) {
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

inline std::string to_yaml(const Controller & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::msg::Controller & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::msg::Controller & msg)
{
  return multi_mode_control_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::msg::Controller>()
{
  return "multi_mode_control_msgs::msg::Controller";
}

template<>
inline const char * name<multi_mode_control_msgs::msg::Controller>()
{
  return "multi_mode_control_msgs/msg/Controller";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::msg::Controller>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::msg::Controller>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<multi_mode_control_msgs::msg::Controller>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__TRAITS_HPP_
