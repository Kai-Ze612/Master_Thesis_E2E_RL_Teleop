// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multi_mode_control_msgs:srv/SetThresholds.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_THRESHOLDS__TRAITS_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_THRESHOLDS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "multi_mode_control_msgs/srv/detail/set_thresholds__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'level1'
// Member 'level2'
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__traits.hpp"

namespace multi_mode_control_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetThresholds_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: level1
  {
    out << "level1: ";
    to_flow_style_yaml(msg.level1, out);
    out << ", ";
  }

  // member: level2
  {
    out << "level2: ";
    to_flow_style_yaml(msg.level2, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetThresholds_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: level1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "level1:\n";
    to_block_style_yaml(msg.level1, out, indentation + 2);
  }

  // member: level2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "level2:\n";
    to_block_style_yaml(msg.level2, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetThresholds_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace multi_mode_control_msgs

namespace rosidl_generator_traits
{

[[deprecated("use multi_mode_control_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const multi_mode_control_msgs::srv::SetThresholds_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::srv::SetThresholds_Request & msg)
{
  return multi_mode_control_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::srv::SetThresholds_Request>()
{
  return "multi_mode_control_msgs::srv::SetThresholds_Request";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::SetThresholds_Request>()
{
  return "multi_mode_control_msgs/srv/SetThresholds_Request";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::SetThresholds_Request>
  : std::integral_constant<bool, has_fixed_size<multi_mode_control_msgs::msg::ContactThresholds>::value> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::SetThresholds_Request>
  : std::integral_constant<bool, has_bounded_size<multi_mode_control_msgs::msg::ContactThresholds>::value> {};

template<>
struct is_message<multi_mode_control_msgs::srv::SetThresholds_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace multi_mode_control_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetThresholds_Response & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetThresholds_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetThresholds_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace multi_mode_control_msgs

namespace rosidl_generator_traits
{

[[deprecated("use multi_mode_control_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const multi_mode_control_msgs::srv::SetThresholds_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::srv::SetThresholds_Response & msg)
{
  return multi_mode_control_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::srv::SetThresholds_Response>()
{
  return "multi_mode_control_msgs::srv::SetThresholds_Response";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::SetThresholds_Response>()
{
  return "multi_mode_control_msgs/srv/SetThresholds_Response";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::SetThresholds_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::SetThresholds_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<multi_mode_control_msgs::srv::SetThresholds_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<multi_mode_control_msgs::srv::SetThresholds>()
{
  return "multi_mode_control_msgs::srv::SetThresholds";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::SetThresholds>()
{
  return "multi_mode_control_msgs/srv/SetThresholds";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::SetThresholds>
  : std::integral_constant<
    bool,
    has_fixed_size<multi_mode_control_msgs::srv::SetThresholds_Request>::value &&
    has_fixed_size<multi_mode_control_msgs::srv::SetThresholds_Response>::value
  >
{
};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::SetThresholds>
  : std::integral_constant<
    bool,
    has_bounded_size<multi_mode_control_msgs::srv::SetThresholds_Request>::value &&
    has_bounded_size<multi_mode_control_msgs::srv::SetThresholds_Response>::value
  >
{
};

template<>
struct is_service<multi_mode_control_msgs::srv::SetThresholds>
  : std::true_type
{
};

template<>
struct is_service_request<multi_mode_control_msgs::srv::SetThresholds_Request>
  : std::true_type
{
};

template<>
struct is_service_response<multi_mode_control_msgs::srv::SetThresholds_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_THRESHOLDS__TRAITS_HPP_
