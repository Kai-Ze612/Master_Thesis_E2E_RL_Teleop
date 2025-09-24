// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:srv/SetGeomProperties.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__SET_GEOM_PROPERTIES__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__SET_GEOM_PROPERTIES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/srv/detail/set_geom_properties__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'properties'
#include "mujoco_ros_msgs/msg/detail/geom_properties__traits.hpp"

namespace mujoco_ros_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetGeomProperties_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: properties
  {
    out << "properties: ";
    to_flow_style_yaml(msg.properties, out);
    out << ", ";
  }

  // member: set_type
  {
    out << "set_type: ";
    rosidl_generator_traits::value_to_yaml(msg.set_type, out);
    out << ", ";
  }

  // member: set_mass
  {
    out << "set_mass: ";
    rosidl_generator_traits::value_to_yaml(msg.set_mass, out);
    out << ", ";
  }

  // member: set_friction
  {
    out << "set_friction: ";
    rosidl_generator_traits::value_to_yaml(msg.set_friction, out);
    out << ", ";
  }

  // member: set_size
  {
    out << "set_size: ";
    rosidl_generator_traits::value_to_yaml(msg.set_size, out);
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
  const SetGeomProperties_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: properties
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "properties:\n";
    to_block_style_yaml(msg.properties, out, indentation + 2);
  }

  // member: set_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "set_type: ";
    rosidl_generator_traits::value_to_yaml(msg.set_type, out);
    out << "\n";
  }

  // member: set_mass
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "set_mass: ";
    rosidl_generator_traits::value_to_yaml(msg.set_mass, out);
    out << "\n";
  }

  // member: set_friction
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "set_friction: ";
    rosidl_generator_traits::value_to_yaml(msg.set_friction, out);
    out << "\n";
  }

  // member: set_size
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "set_size: ";
    rosidl_generator_traits::value_to_yaml(msg.set_size, out);
    out << "\n";
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

inline std::string to_yaml(const SetGeomProperties_Request & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::srv::SetGeomProperties_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::srv::SetGeomProperties_Request & msg)
{
  return mujoco_ros_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::srv::SetGeomProperties_Request>()
{
  return "mujoco_ros_msgs::srv::SetGeomProperties_Request";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::SetGeomProperties_Request>()
{
  return "mujoco_ros_msgs/srv/SetGeomProperties_Request";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::SetGeomProperties_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::SetGeomProperties_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::srv::SetGeomProperties_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace mujoco_ros_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetGeomProperties_Response & msg,
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
  const SetGeomProperties_Response & msg,
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

inline std::string to_yaml(const SetGeomProperties_Response & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::srv::SetGeomProperties_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::srv::SetGeomProperties_Response & msg)
{
  return mujoco_ros_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::srv::SetGeomProperties_Response>()
{
  return "mujoco_ros_msgs::srv::SetGeomProperties_Response";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::SetGeomProperties_Response>()
{
  return "mujoco_ros_msgs/srv/SetGeomProperties_Response";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::SetGeomProperties_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::SetGeomProperties_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::srv::SetGeomProperties_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<mujoco_ros_msgs::srv::SetGeomProperties>()
{
  return "mujoco_ros_msgs::srv::SetGeomProperties";
}

template<>
inline const char * name<mujoco_ros_msgs::srv::SetGeomProperties>()
{
  return "mujoco_ros_msgs/srv/SetGeomProperties";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::srv::SetGeomProperties>
  : std::integral_constant<
    bool,
    has_fixed_size<mujoco_ros_msgs::srv::SetGeomProperties_Request>::value &&
    has_fixed_size<mujoco_ros_msgs::srv::SetGeomProperties_Response>::value
  >
{
};

template<>
struct has_bounded_size<mujoco_ros_msgs::srv::SetGeomProperties>
  : std::integral_constant<
    bool,
    has_bounded_size<mujoco_ros_msgs::srv::SetGeomProperties_Request>::value &&
    has_bounded_size<mujoco_ros_msgs::srv::SetGeomProperties_Response>::value
  >
{
};

template<>
struct is_service<mujoco_ros_msgs::srv::SetGeomProperties>
  : std::true_type
{
};

template<>
struct is_service_request<mujoco_ros_msgs::srv::SetGeomProperties_Request>
  : std::true_type
{
};

template<>
struct is_service_response<mujoco_ros_msgs::srv::SetGeomProperties_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__SET_GEOM_PROPERTIES__TRAITS_HPP_
