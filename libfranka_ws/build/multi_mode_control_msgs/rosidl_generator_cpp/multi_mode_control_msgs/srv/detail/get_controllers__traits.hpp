// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multi_mode_control_msgs:srv/GetControllers.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__GET_CONTROLLERS__TRAITS_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__GET_CONTROLLERS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "multi_mode_control_msgs/srv/detail/get_controllers__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace multi_mode_control_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetControllers_Request & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetControllers_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetControllers_Request & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::srv::GetControllers_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::srv::GetControllers_Request & msg)
{
  return multi_mode_control_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::srv::GetControllers_Request>()
{
  return "multi_mode_control_msgs::srv::GetControllers_Request";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::GetControllers_Request>()
{
  return "multi_mode_control_msgs/srv/GetControllers_Request";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::GetControllers_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::GetControllers_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<multi_mode_control_msgs::srv::GetControllers_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'controllers'
#include "multi_mode_control_msgs/msg/detail/controller__traits.hpp"

namespace multi_mode_control_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetControllers_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: controllers
  {
    if (msg.controllers.size() == 0) {
      out << "controllers: []";
    } else {
      out << "controllers: [";
      size_t pending_items = msg.controllers.size();
      for (auto item : msg.controllers) {
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
  const GetControllers_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: controllers
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.controllers.size() == 0) {
      out << "controllers: []\n";
    } else {
      out << "controllers:\n";
      for (auto item : msg.controllers) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetControllers_Response & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::srv::GetControllers_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::srv::GetControllers_Response & msg)
{
  return multi_mode_control_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::srv::GetControllers_Response>()
{
  return "multi_mode_control_msgs::srv::GetControllers_Response";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::GetControllers_Response>()
{
  return "multi_mode_control_msgs/srv/GetControllers_Response";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::GetControllers_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::GetControllers_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<multi_mode_control_msgs::srv::GetControllers_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<multi_mode_control_msgs::srv::GetControllers>()
{
  return "multi_mode_control_msgs::srv::GetControllers";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::GetControllers>()
{
  return "multi_mode_control_msgs/srv/GetControllers";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::GetControllers>
  : std::integral_constant<
    bool,
    has_fixed_size<multi_mode_control_msgs::srv::GetControllers_Request>::value &&
    has_fixed_size<multi_mode_control_msgs::srv::GetControllers_Response>::value
  >
{
};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::GetControllers>
  : std::integral_constant<
    bool,
    has_bounded_size<multi_mode_control_msgs::srv::GetControllers_Request>::value &&
    has_bounded_size<multi_mode_control_msgs::srv::GetControllers_Response>::value
  >
{
};

template<>
struct is_service<multi_mode_control_msgs::srv::GetControllers>
  : std::true_type
{
};

template<>
struct is_service_request<multi_mode_control_msgs::srv::GetControllers_Request>
  : std::true_type
{
};

template<>
struct is_service_response<multi_mode_control_msgs::srv::GetControllers_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__GET_CONTROLLERS__TRAITS_HPP_
