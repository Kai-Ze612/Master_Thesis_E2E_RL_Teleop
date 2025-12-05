// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:srv/GetGravity.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GRAVITY__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GRAVITY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/srv/detail/get_gravity__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace mujoco_ros_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetGravity_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: admin_hash
  {
    out << "admin_hash: ";
    rosidl_generator_traits::value_to_yaml(msg.admin_hash, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetGravity_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: admin_hash
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "admin_hash: ";
    rosidl_generator_traits::value_to_yaml(msg.admin_hash, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetGravity_Request & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::srv::GetGravity_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::srv::GetGravity_Request & msg)
{
  return mujoco_ros_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::srv::GetGravity_Request>()
{
  return "mujoco_ros_msgs::srv::GetGravity_Request";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::GetGravity_Request>()
{
  return "mujoco_ros_msgs/srv/GetGravity_Request";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::GetGravity_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::GetGravity_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::srv::GetGravity_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace mujoco_ros_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetGravity_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: gravity
  {
    if (msg.gravity.size() == 0) {
      out << "gravity: []";
    } else {
      out << "gravity: [";
      size_t pending_items = msg.gravity.size();
      for (auto item : msg.gravity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: status_message
  {
    out << "status_message: ";
    rosidl_generator_traits::value_to_yaml(msg.status_message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetGravity_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: gravity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.gravity.size() == 0) {
      out << "gravity: []\n";
    } else {
      out << "gravity:\n";
      for (auto item : msg.gravity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: status_message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status_message: ";
    rosidl_generator_traits::value_to_yaml(msg.status_message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetGravity_Response & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::srv::GetGravity_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::srv::GetGravity_Response & msg)
{
  return mujoco_ros_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::srv::GetGravity_Response>()
{
  return "mujoco_ros_msgs::srv::GetGravity_Response";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::GetGravity_Response>()
{
  return "mujoco_ros_msgs/srv/GetGravity_Response";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::GetGravity_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::GetGravity_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::srv::GetGravity_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<mujoco_ros_msgs::srv::GetGravity>()
{
  return "mujoco_ros_msgs::srv::GetGravity";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::GetGravity>()
{
  return "mujoco_ros_msgs/srv/GetGravity";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::GetGravity>
  : std::integral_constant<
    bool,
    has_fixed_size<mujoco_ros_msgs::srv::GetGravity_Request>::value &&
    has_fixed_size<mujoco_ros_msgs::srv::GetGravity_Response>::value
  >
{
};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::GetGravity>
  : std::integral_constant<
    bool,
    has_bounded_size<mujoco_ros_msgs::srv::GetGravity_Request>::value &&
    has_bounded_size<mujoco_ros_msgs::srv::GetGravity_Response>::value
  >
{
};

template<>
struct is_service<mujoco_ros_msgs::srv::GetGravity>
  : std::true_type
{
};

template<>
struct is_service_request<mujoco_ros_msgs::srv::GetGravity_Request>
  : std::true_type
{
};

template<>
struct is_service_response<mujoco_ros_msgs::srv::GetGravity_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GRAVITY__TRAITS_HPP_
