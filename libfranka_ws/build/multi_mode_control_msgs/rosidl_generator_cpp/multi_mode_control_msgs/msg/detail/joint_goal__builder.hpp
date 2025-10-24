// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:msg/JointGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__JOINT_GOAL__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__JOINT_GOAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/msg/detail/joint_goal__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace msg
{

namespace builder
{

class Init_JointGoal_qd
{
public:
  explicit Init_JointGoal_qd(::multi_mode_control_msgs::msg::JointGoal & msg)
  : msg_(msg)
  {}
  ::multi_mode_control_msgs::msg::JointGoal qd(::multi_mode_control_msgs::msg::JointGoal::_qd_type arg)
  {
    msg_.qd = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::JointGoal msg_;
};

class Init_JointGoal_q
{
public:
  Init_JointGoal_q()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointGoal_qd q(::multi_mode_control_msgs::msg::JointGoal::_q_type arg)
  {
    msg_.q = std::move(arg);
    return Init_JointGoal_qd(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::JointGoal msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::msg::JointGoal>()
{
  return multi_mode_control_msgs::msg::builder::Init_JointGoal_q();
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__JOINT_GOAL__BUILDER_HPP_
