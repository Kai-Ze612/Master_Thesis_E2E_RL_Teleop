// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:msg/SimInfo.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/msg/detail/sim_info__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'loading_state'
#include "mujoco_ros_msgs/msg/detail/state_uint__traits.hpp"

namespace mujoco_ros_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const SimInfo & msg,
  std::ostream & out)
{
  out << "{";
  // member: model_path
  {
    out << "model_path: ";
    rosidl_generator_traits::value_to_yaml(msg.model_path, out);
    out << ", ";
  }

  // member: model_valid
  {
    out << "model_valid: ";
    rosidl_generator_traits::value_to_yaml(msg.model_valid, out);
    out << ", ";
  }

  // member: load_count
  {
    out << "load_count: ";
    rosidl_generator_traits::value_to_yaml(msg.load_count, out);
    out << ", ";
  }

  // member: loading_state
  {
    out << "loading_state: ";
    to_flow_style_yaml(msg.loading_state, out);
    out << ", ";
  }

  // member: paused
  {
    out << "paused: ";
    rosidl_generator_traits::value_to_yaml(msg.paused, out);
    out << ", ";
  }

  // member: pending_sim_steps
  {
    out << "pending_sim_steps: ";
    rosidl_generator_traits::value_to_yaml(msg.pending_sim_steps, out);
    out << ", ";
  }

  // member: rt_measured
  {
    out << "rt_measured: ";
    rosidl_generator_traits::value_to_yaml(msg.rt_measured, out);
    out << ", ";
  }

  // member: rt_setting
  {
    out << "rt_setting: ";
    rosidl_generator_traits::value_to_yaml(msg.rt_setting, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SimInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: model_path
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "model_path: ";
    rosidl_generator_traits::value_to_yaml(msg.model_path, out);
    out << "\n";
  }

  // member: model_valid
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "model_valid: ";
    rosidl_generator_traits::value_to_yaml(msg.model_valid, out);
    out << "\n";
  }

  // member: load_count
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "load_count: ";
    rosidl_generator_traits::value_to_yaml(msg.load_count, out);
    out << "\n";
  }

  // member: loading_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "loading_state:\n";
    to_block_style_yaml(msg.loading_state, out, indentation + 2);
  }

  // member: paused
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "paused: ";
    rosidl_generator_traits::value_to_yaml(msg.paused, out);
    out << "\n";
  }

  // member: pending_sim_steps
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pending_sim_steps: ";
    rosidl_generator_traits::value_to_yaml(msg.pending_sim_steps, out);
    out << "\n";
  }

  // member: rt_measured
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rt_measured: ";
    rosidl_generator_traits::value_to_yaml(msg.rt_measured, out);
    out << "\n";
  }

  // member: rt_setting
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rt_setting: ";
    rosidl_generator_traits::value_to_yaml(msg.rt_setting, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SimInfo & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::msg::SimInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::msg::SimInfo & msg)
{
  return mujoco_ros_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::msg::SimInfo>()
{
  return "mujoco_ros_msgs::msg::SimInfo";
}

template<>
inline const char * name<mujoco_ros_msgs::msg::SimInfo>()
{
  return "mujoco_ros_msgs/msg/SimInfo";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::msg::SimInfo>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::msg::SimInfo>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::msg::SimInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__TRAITS_HPP_
