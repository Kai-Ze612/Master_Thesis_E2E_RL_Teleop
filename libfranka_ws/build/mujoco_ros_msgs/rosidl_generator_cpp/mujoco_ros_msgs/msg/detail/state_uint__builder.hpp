// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/StateUint.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__STATE_UINT__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__STATE_UINT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/state_uint__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_StateUint_description
{
public:
  explicit Init_StateUint_description(::mujoco_ros_msgs::msg::StateUint & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::StateUint description(::mujoco_ros_msgs::msg::StateUint::_description_type arg)
  {
    msg_.description = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::StateUint msg_;
};

class Init_StateUint_value
{
public:
  Init_StateUint_value()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_StateUint_description value(::mujoco_ros_msgs::msg::StateUint::_value_type arg)
  {
    msg_.value = std::move(arg);
    return Init_StateUint_description(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::StateUint msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::StateUint>()
{
  return mujoco_ros_msgs::msg::builder::Init_StateUint_value();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__STATE_UINT__BUILDER_HPP_
