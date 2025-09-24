// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from panda_motion_generator_msgs:msg/SetTimeScaling.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__BUILDER_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "panda_motion_generator_msgs/msg/detail/set_time_scaling__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace panda_motion_generator_msgs
{

namespace msg
{

namespace builder
{

class Init_SetTimeScaling_scaling
{
public:
  explicit Init_SetTimeScaling_scaling(::panda_motion_generator_msgs::msg::SetTimeScaling & msg)
  : msg_(msg)
  {}
  ::panda_motion_generator_msgs::msg::SetTimeScaling scaling(::panda_motion_generator_msgs::msg::SetTimeScaling::_scaling_type arg)
  {
    msg_.scaling = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::msg::SetTimeScaling msg_;
};

class Init_SetTimeScaling_steps
{
public:
  Init_SetTimeScaling_steps()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetTimeScaling_scaling steps(::panda_motion_generator_msgs::msg::SetTimeScaling::_steps_type arg)
  {
    msg_.steps = std::move(arg);
    return Init_SetTimeScaling_scaling(msg_);
  }

private:
  ::panda_motion_generator_msgs::msg::SetTimeScaling msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::msg::SetTimeScaling>()
{
  return panda_motion_generator_msgs::msg::builder::Init_SetTimeScaling_steps();
}

}  // namespace panda_motion_generator_msgs

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__BUILDER_HPP_
