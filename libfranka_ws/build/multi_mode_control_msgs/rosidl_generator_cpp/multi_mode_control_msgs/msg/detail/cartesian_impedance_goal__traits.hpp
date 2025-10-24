// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multi_mode_control_msgs:msg/CartesianImpedanceGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CARTESIAN_IMPEDANCE_GOAL__TRAITS_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CARTESIAN_IMPEDANCE_GOAL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "multi_mode_control_msgs/msg/detail/cartesian_impedance_goal__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace multi_mode_control_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const CartesianImpedanceGoal & msg,
  std::ostream & out)
{
  out << "{";
  // member: pose
  {
    out << "pose: ";
    to_flow_style_yaml(msg.pose, out);
    out << ", ";
  }

  // member: q_n
  {
    if (msg.q_n.size() == 0) {
      out << "q_n: []";
    } else {
      out << "q_n: [";
      size_t pending_items = msg.q_n.size();
      for (auto item : msg.q_n) {
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
  const CartesianImpedanceGoal & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pose:\n";
    to_block_style_yaml(msg.pose, out, indentation + 2);
  }

  // member: q_n
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.q_n.size() == 0) {
      out << "q_n: []\n";
    } else {
      out << "q_n:\n";
      for (auto item : msg.q_n) {
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

inline std::string to_yaml(const CartesianImpedanceGoal & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::msg::CartesianImpedanceGoal & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::msg::CartesianImpedanceGoal & msg)
{
  return multi_mode_control_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::msg::CartesianImpedanceGoal>()
{
  return "multi_mode_control_msgs::msg::CartesianImpedanceGoal";
}

template<>
inline const char * name<multi_mode_control_msgs::msg::CartesianImpedanceGoal>()
{
  return "multi_mode_control_msgs/msg/CartesianImpedanceGoal";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::msg::CartesianImpedanceGoal>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Pose>::value> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::msg::CartesianImpedanceGoal>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Pose>::value> {};

template<>
struct is_message<multi_mode_control_msgs::msg::CartesianImpedanceGoal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CARTESIAN_IMPEDANCE_GOAL__TRAITS_HPP_
