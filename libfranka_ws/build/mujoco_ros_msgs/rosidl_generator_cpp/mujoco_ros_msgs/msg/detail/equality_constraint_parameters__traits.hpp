// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mujoco_ros_msgs:msg/EqualityConstraintParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__TRAITS_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mujoco_ros_msgs/msg/detail/equality_constraint_parameters__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'type'
#include "mujoco_ros_msgs/msg/detail/equality_constraint_type__traits.hpp"
// Member 'solver_parameters'
#include "mujoco_ros_msgs/msg/detail/solver_parameters__traits.hpp"
// Member 'anchor'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"
// Member 'relpose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace mujoco_ros_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const EqualityConstraintParameters & msg,
  std::ostream & out)
{
  out << "{";
  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << ", ";
  }

  // member: type
  {
    out << "type: ";
    to_flow_style_yaml(msg.type, out);
    out << ", ";
  }

  // member: solver_parameters
  {
    out << "solver_parameters: ";
    to_flow_style_yaml(msg.solver_parameters, out);
    out << ", ";
  }

  // member: active
  {
    out << "active: ";
    rosidl_generator_traits::value_to_yaml(msg.active, out);
    out << ", ";
  }

  // member: class_param
  {
    out << "class_param: ";
    rosidl_generator_traits::value_to_yaml(msg.class_param, out);
    out << ", ";
  }

  // member: element1
  {
    out << "element1: ";
    rosidl_generator_traits::value_to_yaml(msg.element1, out);
    out << ", ";
  }

  // member: element2
  {
    out << "element2: ";
    rosidl_generator_traits::value_to_yaml(msg.element2, out);
    out << ", ";
  }

  // member: torquescale
  {
    out << "torquescale: ";
    rosidl_generator_traits::value_to_yaml(msg.torquescale, out);
    out << ", ";
  }

  // member: anchor
  {
    out << "anchor: ";
    to_flow_style_yaml(msg.anchor, out);
    out << ", ";
  }

  // member: relpose
  {
    out << "relpose: ";
    to_flow_style_yaml(msg.relpose, out);
    out << ", ";
  }

  // member: polycoef
  {
    if (msg.polycoef.size() == 0) {
      out << "polycoef: []";
    } else {
      out << "polycoef: [";
      size_t pending_items = msg.polycoef.size();
      for (auto item : msg.polycoef) {
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
  const EqualityConstraintParameters & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }

  // member: type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "type:\n";
    to_block_style_yaml(msg.type, out, indentation + 2);
  }

  // member: solver_parameters
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "solver_parameters:\n";
    to_block_style_yaml(msg.solver_parameters, out, indentation + 2);
  }

  // member: active
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "active: ";
    rosidl_generator_traits::value_to_yaml(msg.active, out);
    out << "\n";
  }

  // member: class_param
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "class_param: ";
    rosidl_generator_traits::value_to_yaml(msg.class_param, out);
    out << "\n";
  }

  // member: element1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "element1: ";
    rosidl_generator_traits::value_to_yaml(msg.element1, out);
    out << "\n";
  }

  // member: element2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "element2: ";
    rosidl_generator_traits::value_to_yaml(msg.element2, out);
    out << "\n";
  }

  // member: torquescale
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "torquescale: ";
    rosidl_generator_traits::value_to_yaml(msg.torquescale, out);
    out << "\n";
  }

  // member: anchor
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "anchor:\n";
    to_block_style_yaml(msg.anchor, out, indentation + 2);
  }

  // member: relpose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "relpose:\n";
    to_block_style_yaml(msg.relpose, out, indentation + 2);
  }

  // member: polycoef
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.polycoef.size() == 0) {
      out << "polycoef: []\n";
    } else {
      out << "polycoef:\n";
      for (auto item : msg.polycoef) {
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

inline std::string to_yaml(const EqualityConstraintParameters & msg, bool use_flow_style = false)
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
  const mujoco_ros_msgs::msg::EqualityConstraintParameters & msg,
  std::ostream & out, size_t indentation = 0)
{
  mujoco_ros_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mujoco_ros_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
{
  return mujoco_ros_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mujoco_ros_msgs::msg::EqualityConstraintParameters>()
{
  return "mujoco_ros_msgs::msg::EqualityConstraintParameters";
}

template<>
inline const char * name<mujoco_ros_msgs::msg::EqualityConstraintParameters>()
{
  return "mujoco_ros_msgs/msg/EqualityConstraintParameters";
}

template<>
struct has_fixed_size<mujoco_ros_msgs::msg::EqualityConstraintParameters>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mujoco_ros_msgs::msg::EqualityConstraintParameters>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mujoco_ros_msgs::msg::EqualityConstraintParameters>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__TRAITS_HPP_
