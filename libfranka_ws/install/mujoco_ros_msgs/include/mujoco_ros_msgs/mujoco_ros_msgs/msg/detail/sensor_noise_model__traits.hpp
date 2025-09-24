// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:msg/SensorNoiseModel.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/msg/detail/sensor_noise_model__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace mujoco_ros_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const SensorNoiseModel & msg,
  std::ostream & out)
{
  out << "{";
  // member: sensor_name
  {
    out << "sensor_name: ";
    rosidl_generator_traits::value_to_yaml(msg.sensor_name, out);
    out << ", ";
  }

  // member: mean
  {
    if (msg.mean.size() == 0) {
      out << "mean: []";
    } else {
      out << "mean: [";
      size_t pending_items = msg.mean.size();
      for (auto item : msg.mean) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: std
  {
    if (msg.std.size() == 0) {
      out << "std: []";
    } else {
      out << "std: [";
      size_t pending_items = msg.std.size();
      for (auto item : msg.std) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: set_flag
  {
    out << "set_flag: ";
    rosidl_generator_traits::value_to_yaml(msg.set_flag, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SensorNoiseModel & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: sensor_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sensor_name: ";
    rosidl_generator_traits::value_to_yaml(msg.sensor_name, out);
    out << "\n";
  }

  // member: mean
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.mean.size() == 0) {
      out << "mean: []\n";
    } else {
      out << "mean:\n";
      for (auto item : msg.mean) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: std
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.std.size() == 0) {
      out << "std: []\n";
    } else {
      out << "std:\n";
      for (auto item : msg.std) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: set_flag
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "set_flag: ";
    rosidl_generator_traits::value_to_yaml(msg.set_flag, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SensorNoiseModel & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::msg::SensorNoiseModel & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::msg::SensorNoiseModel & msg)
{
  return mujoco_ros_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::msg::SensorNoiseModel>()
{
  return "mujoco_ros_msgs::msg::SensorNoiseModel";
}

template<>
inline const char * name<mujoco_ros_msgs::msg::SensorNoiseModel>()
{
  return "mujoco_ros_msgs/msg/SensorNoiseModel";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::msg::SensorNoiseModel>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::msg::SensorNoiseModel>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::msg::SensorNoiseModel>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__TRAITS_HPP_
