// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:srv/GetStateUint.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_STATE_UINT__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_STATE_UINT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/srv/detail/get_state_uint__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace mujoco_ros_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetStateUint_Request & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetStateUint_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetStateUint_Request & msg, bool use_flow_style = false)
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

}  // namespace mujoco_ros_msgs

namespace rosidl_generator_traits
{

[[deprecated("use mujoco_ros_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const mujoco_ros_msgs::srv::GetStateUint_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::srv::GetStateUint_Request & msg)
{
  return mujoco_ros_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::srv::GetStateUint_Request>()
{
  return "mujoco_ros_msgs::srv::GetStateUint_Request";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::GetStateUint_Request>()
{
  return "mujoco_ros_msgs/srv/GetStateUint_Request";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::GetStateUint_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::GetStateUint_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<mujoco_ros_msgs::srv::GetStateUint_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'state'
#include "mujoco_ros_msgs/msg/detail/state_uint__traits.hpp"

namespace mujoco_ros_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetStateUint_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: state
  {
    out << "state: ";
    to_flow_style_yaml(msg.state, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetStateUint_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "state:\n";
    to_block_style_yaml(msg.state, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetStateUint_Response & msg, bool use_flow_style = false)
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

}  // namespace mujoco_ros_msgs

namespace rosidl_generator_traits
{

[[deprecated("use mujoco_ros_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const mujoco_ros_msgs::srv::GetStateUint_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::srv::GetStateUint_Response & msg)
{
  return mujoco_ros_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::srv::GetStateUint_Response>()
{
  return "mujoco_ros_msgs::srv::GetStateUint_Response";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::GetStateUint_Response>()
{
  return "mujoco_ros_msgs/srv/GetStateUint_Response";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::GetStateUint_Response>
  : std::integral_constant<bool, has_fixed_size<mujoco_ros_msgs::msg::StateUint>::value> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::GetStateUint_Response>
  : std::integral_constant<bool, has_bounded_size<mujoco_ros_msgs::msg::StateUint>::value> {};

template<>
struct is_message<mujoco_ros_msgs::srv::GetStateUint_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<mujoco_ros_msgs::srv::GetStateUint>()
{
  return "mujoco_ros_msgs::srv::GetStateUint";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::GetStateUint>()
{
  return "mujoco_ros_msgs/srv/GetStateUint";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::GetStateUint>
  : std::integral_constant<
    bool,
    has_fixed_size<mujoco_ros_msgs::srv::GetStateUint_Request>::value &&
    has_fixed_size<mujoco_ros_msgs::srv::GetStateUint_Response>::value
  >
{
};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::GetStateUint>
  : std::integral_constant<
    bool,
    has_bounded_size<mujoco_ros_msgs::srv::GetStateUint_Request>::value &&
    has_bounded_size<mujoco_ros_msgs::srv::GetStateUint_Response>::value
  >
{
};

template<>
struct is_service<mujoco_ros_msgs::srv::GetStateUint>
  : std::true_type
{
};

template<>
struct is_service_request<mujoco_ros_msgs::srv::GetStateUint_Request>
  : std::true_type
{
};

template<>
struct is_service_response<mujoco_ros_msgs::srv::GetStateUint_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_STATE_UINT__TRAITS_HPP_
