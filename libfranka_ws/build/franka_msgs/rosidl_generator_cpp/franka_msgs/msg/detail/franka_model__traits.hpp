// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from franka_msgs:msg/FrankaModel.idl
// generated code does not contain a copyright notice

#ifndef FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__TRAITS_HPP_
#define FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "franka_msgs/msg/detail/franka_model__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace franka_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const FrankaModel & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: coriolis
  {
    if (msg.coriolis.size() == 0) {
      out << "coriolis: []";
    } else {
      out << "coriolis: [";
      size_t pending_items = msg.coriolis.size();
      for (auto item : msg.coriolis) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ee_zero_jacobian
  {
    if (msg.ee_zero_jacobian.size() == 0) {
      out << "ee_zero_jacobian: []";
    } else {
      out << "ee_zero_jacobian: [";
      size_t pending_items = msg.ee_zero_jacobian.size();
      for (auto item : msg.ee_zero_jacobian) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ee_body_jacobian
  {
    if (msg.ee_body_jacobian.size() == 0) {
      out << "ee_body_jacobian: []";
    } else {
      out << "ee_body_jacobian: [";
      size_t pending_items = msg.ee_body_jacobian.size();
      for (auto item : msg.ee_body_jacobian) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: mass
  {
    if (msg.mass.size() == 0) {
      out << "mass: []";
    } else {
      out << "mass: [";
      size_t pending_items = msg.mass.size();
      for (auto item : msg.mass) {
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
  const FrankaModel & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: coriolis
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.coriolis.size() == 0) {
      out << "coriolis: []\n";
    } else {
      out << "coriolis:\n";
      for (auto item : msg.coriolis) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: ee_zero_jacobian
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.ee_zero_jacobian.size() == 0) {
      out << "ee_zero_jacobian: []\n";
    } else {
      out << "ee_zero_jacobian:\n";
      for (auto item : msg.ee_zero_jacobian) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: ee_body_jacobian
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.ee_body_jacobian.size() == 0) {
      out << "ee_body_jacobian: []\n";
    } else {
      out << "ee_body_jacobian:\n";
      for (auto item : msg.ee_body_jacobian) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: mass
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.mass.size() == 0) {
      out << "mass: []\n";
    } else {
      out << "mass:\n";
      for (auto item : msg.mass) {
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

inline std::string to_yaml(const FrankaModel & msg, bool use_flow_style = false)
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

}  // namespace franka_msgs

namespace rosidl_generator_traits
{

[[deprecated("use franka_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const franka_msgs::msg::FrankaModel & msg,
  std::ostream & out, size_t indentation = 0)
{
  franka_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use franka_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const franka_msgs::msg::FrankaModel & msg)
{
  return franka_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<franka_msgs::msg::FrankaModel>()
{
  return "franka_msgs::msg::FrankaModel";
}

template<>
inline const char * name<franka_msgs::msg::FrankaModel>()
{
  return "franka_msgs/msg/FrankaModel";
}

template<>
struct has_fixed_size<franka_msgs::msg::FrankaModel>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<franka_msgs::msg::FrankaModel>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<franka_msgs::msg::FrankaModel>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__TRAITS_HPP_
