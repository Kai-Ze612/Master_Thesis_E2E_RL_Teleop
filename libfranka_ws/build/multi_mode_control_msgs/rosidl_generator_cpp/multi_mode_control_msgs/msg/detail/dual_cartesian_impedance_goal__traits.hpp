// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multi_mode_control_msgs:msg/DualCartesianImpedanceGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__TRAITS_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "multi_mode_control_msgs/msg/detail/dual_cartesian_impedance_goal__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'l_pose'
// Member 'r_pose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace multi_mode_control_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const DualCartesianImpedanceGoal & msg,
  std::ostream & out)
{
  out << "{";
  // member: l_pose
  {
    out << "l_pose: ";
    to_flow_style_yaml(msg.l_pose, out);
    out << ", ";
  }

  // member: r_pose
  {
    out << "r_pose: ";
    to_flow_style_yaml(msg.r_pose, out);
    out << ", ";
  }

  // member: l_q_n
  {
    if (msg.l_q_n.size() == 0) {
      out << "l_q_n: []";
    } else {
      out << "l_q_n: [";
      size_t pending_items = msg.l_q_n.size();
      for (auto item : msg.l_q_n) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: r_q_n
  {
    if (msg.r_q_n.size() == 0) {
      out << "r_q_n: []";
    } else {
      out << "r_q_n: [";
      size_t pending_items = msg.r_q_n.size();
      for (auto item : msg.r_q_n) {
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
  const DualCartesianImpedanceGoal & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: l_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "l_pose:\n";
    to_block_style_yaml(msg.l_pose, out, indentation + 2);
  }

  // member: r_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "r_pose:\n";
    to_block_style_yaml(msg.r_pose, out, indentation + 2);
  }

  // member: l_q_n
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.l_q_n.size() == 0) {
      out << "l_q_n: []\n";
    } else {
      out << "l_q_n:\n";
      for (auto item : msg.l_q_n) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: r_q_n
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.r_q_n.size() == 0) {
      out << "r_q_n: []\n";
    } else {
      out << "r_q_n:\n";
      for (auto item : msg.r_q_n) {
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

inline std::string to_yaml(const DualCartesianImpedanceGoal & msg, bool use_flow_style = false)
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
  const multi_mode_control_msgs::msg::DualCartesianImpedanceGoal & msg,
  std::ostream & out, size_t indentation = 0)
{
  multi_mode_control_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use multi_mode_control_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const multi_mode_control_msgs::msg::DualCartesianImpedanceGoal & msg)
{
  return multi_mode_control_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal>()
{
  return "multi_mode_control_msgs::msg::DualCartesianImpedanceGoal";
}

template<>
inline const char * name<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal>()
{
  return "multi_mode_control_msgs/msg/DualCartesianImpedanceGoal";
}

template<>
struct has_fixed_size<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Pose>::value> {};

template<>
struct has_bounded_size<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Pose>::value> {};

template<>
struct is_message<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__TRAITS_HPP_
