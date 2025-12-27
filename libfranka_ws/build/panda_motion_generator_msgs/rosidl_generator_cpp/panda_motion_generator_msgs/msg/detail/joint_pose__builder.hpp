// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from panda_motion_generator_msgs:msg/JointPose.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__BUILDER_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "panda_motion_generator_msgs/msg/detail/joint_pose__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace panda_motion_generator_msgs
{

namespace msg
{

namespace builder
{

class Init_JointPose_q
{
public:
  Init_JointPose_q()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::panda_motion_generator_msgs::msg::JointPose q(::panda_motion_generator_msgs::msg::JointPose::_q_type arg)
  {
    msg_.q = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::msg::JointPose msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::msg::JointPose>()
{
  return panda_motion_generator_msgs::msg::builder::Init_JointPose_q();
}

}  // namespace panda_motion_generator_msgs

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__BUILDER_HPP_
