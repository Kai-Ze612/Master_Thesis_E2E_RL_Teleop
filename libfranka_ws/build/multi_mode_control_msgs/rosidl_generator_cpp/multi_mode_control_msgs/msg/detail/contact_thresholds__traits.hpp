// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multi_mode_control_msgs:msg/ContactThresholds.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__TRAITS_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "multi_mode_control_msgs/msg/detail/contact_thresholds__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'tau_max'
// Member 'tau_min'
#include "multi_mode_control_msgs/msg/detail/joint_array__traits.hpp"

namespace multi_mode_control_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ContactThresholds & msg,
  std::ostream & out)
{
  out << "{";
  // member: f_max
  {
    if (msg.f_max.size() == 0) {
      out << "f_max: []";
    } else {
      out << "f_max: [";
      size_t pending_items = msg.f_max.size();
      for (auto item : msg.f_max) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_min
  {
    if (msg.f_min.size() == 0) {
      out << "f_min: []";
    } else {
      out << "f_min: [";
      size_t pending_items = msg.f_min.size();
      for (auto item : msg.f_min) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: tau_max
  {
    if (msg.tau_max.size() == 0) {
      out << "tau_max: []";
    } else {
      out << "tau_max: [";
      size_t pending_items = msg.tau_max.size();
      for (auto item : msg.tau_max) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: tau_min
  {
    if (msg.tau_min.size() == 0) {
      out << "tau_min: []";
    } else {
      out << "tau_min: [";
      size_t pending_items = msg.tau_min.size();
      for (auto item : msg.tau_min) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_abs_max
  {
    out << "f_abs_max: ";
    rosidl_generator_traits::value_to_yaml(msg.f_abs_max, out);
    out << ", ";
  }

  // member: m_abs_max
  {
    out << "m_abs_max: ";
    rosidl_generator_traits::value_to_yaml(msg.m_abs_max, out);
    out << ", ";
  }

  // member: abs_mode
  {
    out << "abs_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.abs_mode, out);
    out << ", ";
  }

  // member: directional_mode
  {
    out << "directional_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.directional_mode, out);
    out << ", ";
  }

  // member: torque_mode
  {
    out << "torque_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.torque_mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ContactThresholds & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: f_max
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_max.size() == 0) {
      out << "f_max: []\n";
    } else {
      out << "f_max:\n";
      for (auto item : msg.f_max) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_min
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_min.size() == 0) {
      out << "f_min: []\n";
    } else {
      out << "f_min:\n";
      for (auto item : msg.f_min) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: tau_max
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.tau_max.size() == 0) {
      out << "tau_max: []\n";
    } else {
      out << "tau_max:\n";
      for (auto item : msg.tau_max) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: tau_min
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.tau_min.size() == 0) {
      out << "tau_min: []\n";
    } else {
      out << "tau_min:\n";
      for (auto item : msg.tau_min) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: f_abs_max
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "f_abs_max: ";
    rosidl_generator_traits::value_to_yaml(msg.f_abs_max, out);
    out << "\n";
  }

  // member: m_abs_max
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "m_abs_max: ";
    rosidl_generator_traits::value_to_yaml(msg.m_abs_max, out);
    out << "\n";
  }

  // member: abs_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "abs_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.abs_mode, out);
    out << "\n";
  }

  // member: directional_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "directional_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.directional_mode, out);
    out << "\n";
  }

  // member: torque_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "torque_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.torque_mode, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ContactThresholds & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::msg::ContactThresholds & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::msg::ContactThresholds & msg)
{
  return multi_mode_control_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::msg::ContactThresholds>()
{
  return "multi_mode_control_msgs::msg::ContactThresholds";
}

template<>
inline const char * name<multi_mode_control_msgs::msg::ContactThresholds>()
{
  return "multi_mode_control_msgs/msg/ContactThresholds";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::msg::ContactThresholds>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::msg::ContactThresholds>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<multi_mode_control_msgs::msg::ContactThresholds>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__TRAITS_HPP_
