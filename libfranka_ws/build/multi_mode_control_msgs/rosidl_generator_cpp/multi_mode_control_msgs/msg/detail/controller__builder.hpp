// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:msg/Controller.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/msg/detail/controller__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace msg
{

namespace builder
{

class Init_Controller_resources
{
public:
  explicit Init_Controller_resources(::multi_mode_control_msgs::msg::Controller & msg)
  : msg_(msg)
  {}
  ::multi_mode_control_msgs::msg::Controller resources(::multi_mode_control_msgs::msg::Controller::_resources_type arg)
  {
    msg_.resources = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::Controller msg_;
};

class Init_Controller_name
{
public:
  Init_Controller_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Controller_resources name(::multi_mode_control_msgs::msg::Controller::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_Controller_resources(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::Controller msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::msg::Controller>()
{
  return multi_mode_control_msgs::msg::builder::Init_Controller_name();
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTROLLER__BUILDER_HPP_
