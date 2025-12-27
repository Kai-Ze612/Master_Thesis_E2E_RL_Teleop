// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:msg/CartesianImpedanceGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CARTESIAN_IMPEDANCE_GOAL__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CARTESIAN_IMPEDANCE_GOAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/msg/detail/cartesian_impedance_goal__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace msg
{

namespace builder
{

class Init_CartesianImpedanceGoal_q_n
{
public:
  explicit Init_CartesianImpedanceGoal_q_n(::multi_mode_control_msgs::msg::CartesianImpedanceGoal & msg)
  : msg_(msg)
  {}
  ::multi_mode_control_msgs::msg::CartesianImpedanceGoal q_n(::multi_mode_control_msgs::msg::CartesianImpedanceGoal::_q_n_type arg)
  {
    msg_.q_n = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::CartesianImpedanceGoal msg_;
};

class Init_CartesianImpedanceGoal_pose
{
public:
  Init_CartesianImpedanceGoal_pose()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CartesianImpedanceGoal_q_n pose(::multi_mode_control_msgs::msg::CartesianImpedanceGoal::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return Init_CartesianImpedanceGoal_q_n(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::CartesianImpedanceGoal msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::msg::CartesianImpedanceGoal>()
{
  return multi_mode_control_msgs::msg::builder::Init_CartesianImpedanceGoal_pose();
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CARTESIAN_IMPEDANCE_GOAL__BUILDER_HPP_
