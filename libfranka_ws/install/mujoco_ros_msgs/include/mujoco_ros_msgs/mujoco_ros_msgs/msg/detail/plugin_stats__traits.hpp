// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:msg/PluginStats.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/msg/detail/plugin_stats__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace mujoco_ros_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PluginStats & msg,
  std::ostream & out)
{
  out << "{";
  // member: plugin_type
  {
    out << "plugin_type: ";
    rosidl_generator_traits::value_to_yaml(msg.plugin_type, out);
    out << ", ";
  }

  // member: load_time
  {
    out << "load_time: ";
    rosidl_generator_traits::value_to_yaml(msg.load_time, out);
    out << ", ";
  }

  // member: reset_time
  {
    out << "reset_time: ";
    rosidl_generator_traits::value_to_yaml(msg.reset_time, out);
    out << ", ";
  }

  // member: ema_steptime_control
  {
    out << "ema_steptime_control: ";
    rosidl_generator_traits::value_to_yaml(msg.ema_steptime_control, out);
    out << ", ";
  }

  // member: ema_steptime_passive
  {
    out << "ema_steptime_passive: ";
    rosidl_generator_traits::value_to_yaml(msg.ema_steptime_passive, out);
    out << ", ";
  }

  // member: ema_steptime_render
  {
    out << "ema_steptime_render: ";
    rosidl_generator_traits::value_to_yaml(msg.ema_steptime_render, out);
    out << ", ";
  }

  // member: ema_steptime_last_stage
  {
    out << "ema_steptime_last_stage: ";
    rosidl_generator_traits::value_to_yaml(msg.ema_steptime_last_stage, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PluginStats & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: plugin_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "plugin_type: ";
    rosidl_generator_traits::value_to_yaml(msg.plugin_type, out);
    out << "\n";
  }

  // member: load_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "load_time: ";
    rosidl_generator_traits::value_to_yaml(msg.load_time, out);
    out << "\n";
  }

  // member: reset_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reset_time: ";
    rosidl_generator_traits::value_to_yaml(msg.reset_time, out);
    out << "\n";
  }

  // member: ema_steptime_control
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ema_steptime_control: ";
    rosidl_generator_traits::value_to_yaml(msg.ema_steptime_control, out);
    out << "\n";
  }

  // member: ema_steptime_passive
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ema_steptime_passive: ";
    rosidl_generator_traits::value_to_yaml(msg.ema_steptime_passive, out);
    out << "\n";
  }

  // member: ema_steptime_render
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ema_steptime_render: ";
    rosidl_generator_traits::value_to_yaml(msg.ema_steptime_render, out);
    out << "\n";
  }

  // member: ema_steptime_last_stage
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ema_steptime_last_stage: ";
    rosidl_generator_traits::value_to_yaml(msg.ema_steptime_last_stage, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PluginStats & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::msg::PluginStats & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::msg::PluginStats & msg)
{
  return mujoco_ros_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::msg::PluginStats>()
{
  return "mujoco_ros_msgs::msg::PluginStats";
}

template<>
inline const char * name<mujoco_ros_msgs::msg::PluginStats>()
{
  return "mujoco_ros_msgs/msg/PluginStats";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::msg::PluginStats>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::msg::PluginStats>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::msg::PluginStats>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__TRAITS_HPP_
