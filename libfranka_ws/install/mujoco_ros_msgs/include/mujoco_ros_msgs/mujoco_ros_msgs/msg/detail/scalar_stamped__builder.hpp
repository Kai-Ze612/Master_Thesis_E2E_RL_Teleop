// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/ScalarStamped.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SCALAR_STAMPED__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SCALAR_STAMPED__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/scalar_stamped__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_ScalarStamped_value
{
public:
  explicit Init_ScalarStamped_value(::mujoco_ros_msgs::msg::ScalarStamped & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::ScalarStamped value(::mujoco_ros_msgs::msg::ScalarStamped::_value_type arg)
  {
    msg_.value = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::ScalarStamped msg_;
};

class Init_ScalarStamped_header
{
public:
  Init_ScalarStamped_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ScalarStamped_value header(::mujoco_ros_msgs::msg::ScalarStamped::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ScalarStamped_value(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::ScalarStamped msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::ScalarStamped>()
{
  return mujoco_ros_msgs::msg::builder::Init_ScalarStamped_header();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SCALAR_STAMPED__BUILDER_HPP_
