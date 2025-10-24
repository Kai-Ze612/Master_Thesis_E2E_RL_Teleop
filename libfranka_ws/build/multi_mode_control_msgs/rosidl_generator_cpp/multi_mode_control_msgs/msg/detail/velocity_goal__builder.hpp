// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:msg/VelocityGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__VELOCITY_GOAL__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__VELOCITY_GOAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/msg/detail/velocity_goal__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace msg
{

namespace builder
{

class Init_VelocityGoal_qd
{
public:
  Init_VelocityGoal_qd()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::multi_mode_control_msgs::msg::VelocityGoal qd(::multi_mode_control_msgs::msg::VelocityGoal::_qd_type arg)
  {
    msg_.qd = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::VelocityGoal msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::msg::VelocityGoal>()
{
  return multi_mode_control_msgs::msg::builder::Init_VelocityGoal_qd();
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__VELOCITY_GOAL__BUILDER_HPP_
