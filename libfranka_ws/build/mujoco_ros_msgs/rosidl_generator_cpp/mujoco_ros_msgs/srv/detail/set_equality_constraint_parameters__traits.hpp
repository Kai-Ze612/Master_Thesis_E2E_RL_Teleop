// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:srv/SetEqualityConstraintParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__SET_EQUALITY_CONSTRAINT_PARAMETERS__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__SET_EQUALITY_CONSTRAINT_PARAMETERS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/srv/detail/set_equality_constraint_parameters__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'parameters'
#include "mujoco_ros_msgs/msg/detail/equality_constraint_parameters__traits.hpp"

namespace mujoco_ros_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetEqualityConstraintParameters_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: parameters
  {
    if (msg.parameters.size() == 0) {
      out << "parameters: []";
    } else {
      out << "parameters: [";
      size_t pending_items = msg.parameters.size();
      for (auto item : msg.parameters) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: admin_hash
  {
    out << "admin_hash: ";
    rosidl_generator_traits::value_to_yaml(msg.admin_hash, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetEqualityConstraintParameters_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: parameters
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.parameters.size() == 0) {
      out << "parameters: []\n";
    } else {
      out << "parameters:\n";
      for (auto item : msg.parameters) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

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

inline std::string to_yaml(const SetEqualityConstraintParameters_Request & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request & msg)
{
  return mujoco_ros_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request>()
{
  return "mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request>()
{
  return "mujoco_ros_msgs/srv/SetEqualityConstraintParameters_Request";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace mujoco_ros_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetEqualityConstraintParameters_Response & msg,
  std::ostream & out)
{
  out << "{";
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
  const SetEqualityConstraintParameters_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
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

inline std::string to_yaml(const SetEqualityConstraintParameters_Response & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response & msg)
{
  return mujoco_ros_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response>()
{
  return "mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response>()
{
  return "mujoco_ros_msgs/srv/SetEqualityConstraintParameters_Response";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<mujoco_ros_msgs::srv::SetEqualityConstraintParameters>()
{
  return "mujoco_ros_msgs::srv::SetEqualityConstraintParameters";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::SetEqualityConstraintParameters>()
{
  return "mujoco_ros_msgs/srv/SetEqualityConstraintParameters";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters>
  : std::integral_constant<
    bool,
    has_fixed_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request>::value &&
    has_fixed_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response>::value
  >
{
};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters>
  : std::integral_constant<
    bool,
    has_bounded_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request>::value &&
    has_bounded_size<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response>::value
  >
{
};

template<>
struct is_service<mujoco_ros_msgs::srv::SetEqualityConstraintParameters>
  : std::true_type
{
};

template<>
struct is_service_request<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Request>
  : std::true_type
{
};

template<>
struct is_service_response<mujoco_ros_msgs::srv::SetEqualityConstraintParameters_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__SET_EQUALITY_CONSTRAINT_PARAMETERS__TRAITS_HPP_
