// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multi_mode_control_msgs:srv/SetCartesianImpedance.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_CARTESIAN_IMPEDANCE__TRAITS_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_CARTESIAN_IMPEDANCE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "multi_mode_control_msgs/srv/detail/set_cartesian_impedance__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace multi_mode_control_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetCartesianImpedance_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: stiffness
  {
    if (msg.stiffness.size() == 0) {
      out << "stiffness: []";
    } else {
      out << "stiffness: [";
      size_t pending_items = msg.stiffness.size();
      for (auto item : msg.stiffness) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: damping_ratio
  {
    if (msg.damping_ratio.size() == 0) {
      out << "damping_ratio: []";
    } else {
      out << "damping_ratio: [";
      size_t pending_items = msg.damping_ratio.size();
      for (auto item : msg.damping_ratio) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: nullspace_stiffness
  {
    out << "nullspace_stiffness: ";
    rosidl_generator_traits::value_to_yaml(msg.nullspace_stiffness, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetCartesianImpedance_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stiffness
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.stiffness.size() == 0) {
      out << "stiffness: []\n";
    } else {
      out << "stiffness:\n";
      for (auto item : msg.stiffness) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: damping_ratio
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.damping_ratio.size() == 0) {
      out << "damping_ratio: []\n";
    } else {
      out << "damping_ratio:\n";
      for (auto item : msg.damping_ratio) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: nullspace_stiffness
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "nullspace_stiffness: ";
    rosidl_generator_traits::value_to_yaml(msg.nullspace_stiffness, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetCartesianImpedance_Request & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::srv::SetCartesianImpedance_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::srv::SetCartesianImpedance_Request & msg)
{
  return multi_mode_control_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::srv::SetCartesianImpedance_Request>()
{
  return "multi_mode_control_msgs::srv::SetCartesianImpedance_Request";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::SetCartesianImpedance_Request>()
{
  return "multi_mode_control_msgs/srv/SetCartesianImpedance_Request";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::SetCartesianImpedance_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::SetCartesianImpedance_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<multi_mode_control_msgs::srv::SetCartesianImpedance_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace multi_mode_control_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetCartesianImpedance_Response & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetCartesianImpedance_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetCartesianImpedance_Response & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::srv::SetCartesianImpedance_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::srv::SetCartesianImpedance_Response & msg)
{
  return multi_mode_control_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::srv::SetCartesianImpedance_Response>()
{
  return "multi_mode_control_msgs::srv::SetCartesianImpedance_Response";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::SetCartesianImpedance_Response>()
{
  return "multi_mode_control_msgs/srv/SetCartesianImpedance_Response";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::SetCartesianImpedance_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::SetCartesianImpedance_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<multi_mode_control_msgs::srv::SetCartesianImpedance_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<multi_mode_control_msgs::srv::SetCartesianImpedance>()
{
  return "multi_mode_control_msgs::srv::SetCartesianImpedance";
}

template<>
inline const char * name<multi_mode_control_msgs::srv::SetCartesianImpedance>()
{
  return "multi_mode_control_msgs/srv/SetCartesianImpedance";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::srv::SetCartesianImpedance>
  : std::integral_constant<
    bool,
    has_fixed_size<multi_mode_control_msgs::srv::SetCartesianImpedance_Request>::value &&
    has_fixed_size<multi_mode_control_msgs::srv::SetCartesianImpedance_Response>::value
  >
{
};

template<>
struct has_bounded_size<multi_mode_control_msgs::srv::SetCartesianImpedance>
  : std::integral_constant<
    bool,
    has_bounded_size<multi_mode_control_msgs::srv::SetCartesianImpedance_Request>::value &&
    has_bounded_size<multi_mode_control_msgs::srv::SetCartesianImpedance_Response>::value
  >
{
};

template<>
struct is_service<multi_mode_control_msgs::srv::SetCartesianImpedance>
  : std::true_type
{
};

template<>
struct is_service_request<multi_mode_control_msgs::srv::SetCartesianImpedance_Request>
  : std::true_type
{
};

template<>
struct is_service_response<multi_mode_control_msgs::srv::SetCartesianImpedance_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_CARTESIAN_IMPEDANCE__TRAITS_HPP_
