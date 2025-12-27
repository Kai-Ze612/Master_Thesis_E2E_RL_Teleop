// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:msg/SolverParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/msg/detail/solver_parameters__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace mujoco_ros_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const SolverParameters & msg,
  std::ostream & out)
{
  out << "{";
  // member: dmin
  {
    out << "dmin: ";
    rosidl_generator_traits::value_to_yaml(msg.dmin, out);
    out << ", ";
  }

  // member: dmax
  {
    out << "dmax: ";
    rosidl_generator_traits::value_to_yaml(msg.dmax, out);
    out << ", ";
  }

  // member: width
  {
    out << "width: ";
    rosidl_generator_traits::value_to_yaml(msg.width, out);
    out << ", ";
  }

  // member: midpoint
  {
    out << "midpoint: ";
    rosidl_generator_traits::value_to_yaml(msg.midpoint, out);
    out << ", ";
  }

  // member: power
  {
    out << "power: ";
    rosidl_generator_traits::value_to_yaml(msg.power, out);
    out << ", ";
  }

  // member: timeconst
  {
    out << "timeconst: ";
    rosidl_generator_traits::value_to_yaml(msg.timeconst, out);
    out << ", ";
  }

  // member: dampratio
  {
    out << "dampratio: ";
    rosidl_generator_traits::value_to_yaml(msg.dampratio, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SolverParameters & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: dmin
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dmin: ";
    rosidl_generator_traits::value_to_yaml(msg.dmin, out);
    out << "\n";
  }

  // member: dmax
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dmax: ";
    rosidl_generator_traits::value_to_yaml(msg.dmax, out);
    out << "\n";
  }

  // member: width
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "width: ";
    rosidl_generator_traits::value_to_yaml(msg.width, out);
    out << "\n";
  }

  // member: midpoint
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "midpoint: ";
    rosidl_generator_traits::value_to_yaml(msg.midpoint, out);
    out << "\n";
  }

  // member: power
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "power: ";
    rosidl_generator_traits::value_to_yaml(msg.power, out);
    out << "\n";
  }

  // member: timeconst
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timeconst: ";
    rosidl_generator_traits::value_to_yaml(msg.timeconst, out);
    out << "\n";
  }

  // member: dampratio
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dampratio: ";
    rosidl_generator_traits::value_to_yaml(msg.dampratio, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SolverParameters & msg, bool use_flow_style = false)
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

}  // namespace mujoco_ros_msgs

namespace rosidl_generator_traits
{

[[deprecated("use mujoco_ros_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const mujoco_ros_msgs::msg::SolverParameters & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::msg::SolverParameters & msg)
{
  return mujoco_ros_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::msg::SolverParameters>()
{
  return "mujoco_ros_msgs::msg::SolverParameters";
}

template<>
inline const char * name<mujoco_ros_msgs::msg::SolverParameters>()
{
  return "mujoco_ros_msgs/msg/SolverParameters";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::msg::SolverParameters>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::msg::SolverParameters>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<mujoco_ros_msgs::msg::SolverParameters>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__TRAITS_HPP_
