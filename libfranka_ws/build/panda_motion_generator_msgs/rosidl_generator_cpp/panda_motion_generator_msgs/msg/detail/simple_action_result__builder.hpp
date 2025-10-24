// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from panda_motion_generator_msgs:msg/SimpleActionResult.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__BUILDER_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "panda_motion_generator_msgs/msg/detail/simple_action_result__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace panda_motion_generator_msgs
{

namespace msg
{

namespace builder
{

class Init_SimpleActionResult_error
{
public:
  explicit Init_SimpleActionResult_error(::panda_motion_generator_msgs::msg::SimpleActionResult & msg)
  : msg_(msg)
  {}
  ::panda_motion_generator_msgs::msg::SimpleActionResult error(::panda_motion_generator_msgs::msg::SimpleActionResult::_error_type arg)
  {
    msg_.error = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::msg::SimpleActionResult msg_;
};

class Init_SimpleActionResult_state
{
public:
  Init_SimpleActionResult_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SimpleActionResult_error state(::panda_motion_generator_msgs::msg::SimpleActionResult::_state_type arg)
  {
    msg_.state = std::move(arg);
    return Init_SimpleActionResult_error(msg_);
  }

private:
  ::panda_motion_generator_msgs::msg::SimpleActionResult msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::msg::SimpleActionResult>()
{
  return panda_motion_generator_msgs::msg::builder::Init_SimpleActionResult_state();
}

}  // namespace panda_motion_generator_msgs

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__BUILDER_HPP_
