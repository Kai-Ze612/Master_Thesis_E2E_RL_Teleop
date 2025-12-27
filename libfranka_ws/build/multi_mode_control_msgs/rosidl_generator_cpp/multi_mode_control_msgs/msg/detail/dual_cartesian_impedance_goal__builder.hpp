// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:msg/DualCartesianImpedanceGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/msg/detail/dual_cartesian_impedance_goal__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace msg
{

namespace builder
{

class Init_DualCartesianImpedanceGoal_r_q_n
{
public:
  explicit Init_DualCartesianImpedanceGoal_r_q_n(::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal & msg)
  : msg_(msg)
  {}
  ::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal r_q_n(::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal::_r_q_n_type arg)
  {
    msg_.r_q_n = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal msg_;
};

class Init_DualCartesianImpedanceGoal_l_q_n
{
public:
  explicit Init_DualCartesianImpedanceGoal_l_q_n(::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal & msg)
  : msg_(msg)
  {}
  Init_DualCartesianImpedanceGoal_r_q_n l_q_n(::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal::_l_q_n_type arg)
  {
    msg_.l_q_n = std::move(arg);
    return Init_DualCartesianImpedanceGoal_r_q_n(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal msg_;
};

class Init_DualCartesianImpedanceGoal_r_pose
{
public:
  explicit Init_DualCartesianImpedanceGoal_r_pose(::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal & msg)
  : msg_(msg)
  {}
  Init_DualCartesianImpedanceGoal_l_q_n r_pose(::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal::_r_pose_type arg)
  {
    msg_.r_pose = std::move(arg);
    return Init_DualCartesianImpedanceGoal_l_q_n(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal msg_;
};

class Init_DualCartesianImpedanceGoal_l_pose
{
public:
  Init_DualCartesianImpedanceGoal_l_pose()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DualCartesianImpedanceGoal_r_pose l_pose(::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal::_l_pose_type arg)
  {
    msg_.l_pose = std::move(arg);
    return Init_DualCartesianImpedanceGoal_r_pose(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::msg::DualCartesianImpedanceGoal>()
{
  return multi_mode_control_msgs::msg::builder::Init_DualCartesianImpedanceGoal_l_pose();
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__BUILDER_HPP_
