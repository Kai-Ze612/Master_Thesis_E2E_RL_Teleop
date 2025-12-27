// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multi_mode_control_msgs:srv/SetWorldState.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_WORLD_STATE__TRAITS_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_WORLD_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "multi_mode_control_msgs/srv/detail/set_world_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace multi_mode_control_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetWorldState_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: items
  {
    if (msg.items.size() == 0) {
      out << "items: []";
    } else {
      out << "items: [";
      size_t pending_items = msg.items.size();
      for (auto item : msg.items) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: values
  {
    if (msg.values.size() == 0) {
      out << "values: []";
    } else {
      out << "values: [";
      size_t pending_items = msg.values.size();
      for (auto item : msg.values) {
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
  const SetWorldState_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: items
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.items.size() == 0) {
      out << "items: []\n";
    } else {
      out << "items:\n";
      for (auto item : msg.items) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: values
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.values.size() == 0) {
      out << "values: []\n";
    } else {
      out << "values:\n";
      for (auto item : msg.values) {
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

inline std::string to_yaml(const SetWorldState_Request & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::srv::SetWorldState_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::srv::SetWorldState_Request & msg)
{
  return multi_mode_control_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::srv::SetWorldState_Request>()
{
  return "multi_mode_control_msgs::srv::SetWorldState_Request";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::SetWorldState_Request>()
{
  return "multi_mode_control_msgs/srv/SetWorldState_Request";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::SetWorldState_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::SetWorldState_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<multi_mode_control_msgs::srv::SetWorldState_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace multi_mode_control_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetWorldState_Response & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetWorldState_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetWorldState_Response & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::srv::SetWorldState_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::srv::SetWorldState_Response & msg)
{
  return multi_mode_control_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::srv::SetWorldState_Response>()
{
  return "multi_mode_control_msgs::srv::SetWorldState_Response";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::SetWorldState_Response>()
{
  return "multi_mode_control_msgs/srv/SetWorldState_Response";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::SetWorldState_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::SetWorldState_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<multi_mode_control_msgs::srv::SetWorldState_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<multi_mode_control_msgs::srv::SetWorldState>()
{
  return "multi_mode_control_msgs::srv::SetWorldState";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::SetWorldState>()
{
  return "multi_mode_control_msgs/srv/SetWorldState";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::SetWorldState>
  : std::integral_constant<
    bool,
    has_fixed_size<multi_mode_control_msgs::srv::SetWorldState_Request>::value &&
    has_fixed_size<multi_mode_control_msgs::srv::SetWorldState_Response>::value
  >
{
};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::SetWorldState>
  : std::integral_constant<
    bool,
    has_bounded_size<multi_mode_control_msgs::srv::SetWorldState_Request>::value &&
    has_bounded_size<multi_mode_control_msgs::srv::SetWorldState_Response>::value
  >
{
};

template<>
struct is_service<multi_mode_control_msgs::srv::SetWorldState>
  : std::true_type
{
};

template<>
struct is_service_request<multi_mode_control_msgs::srv::SetWorldState_Request>
  : std::true_type
{
};

template<>
struct is_service_response<multi_mode_control_msgs::srv::SetWorldState_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_WORLD_STATE__TRAITS_HPP_
