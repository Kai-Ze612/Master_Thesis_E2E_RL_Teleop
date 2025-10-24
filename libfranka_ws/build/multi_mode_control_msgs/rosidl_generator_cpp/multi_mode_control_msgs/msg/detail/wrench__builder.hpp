// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:msg/Wrench.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__WRENCH__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__WRENCH__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/msg/detail/wrench__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace msg
{

namespace builder
{

class Init_Wrench_data
{
public:
  Init_Wrench_data()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::multi_mode_control_msgs::msg::Wrench data(::multi_mode_control_msgs::msg::Wrench::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::Wrench msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::msg::Wrench>()
{
  return multi_mode_control_msgs::msg::builder::Init_Wrench_data();
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__WRENCH__BUILDER_HPP_
