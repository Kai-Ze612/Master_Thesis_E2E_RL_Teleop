// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:msg/ContactThresholds.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/msg/detail/contact_thresholds__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace msg
{

namespace builder
{

class Init_ContactThresholds_torque_mode
{
public:
  explicit Init_ContactThresholds_torque_mode(::multi_mode_control_msgs::msg::ContactThresholds & msg)
  : msg_(msg)
  {}
  ::multi_mode_control_msgs::msg::ContactThresholds torque_mode(::multi_mode_control_msgs::msg::ContactThresholds::_torque_mode_type arg)
  {
    msg_.torque_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::ContactThresholds msg_;
};

class Init_ContactThresholds_directional_mode
{
public:
  explicit Init_ContactThresholds_directional_mode(::multi_mode_control_msgs::msg::ContactThresholds & msg)
  : msg_(msg)
  {}
  Init_ContactThresholds_torque_mode directional_mode(::multi_mode_control_msgs::msg::ContactThresholds::_directional_mode_type arg)
  {
    msg_.directional_mode = std::move(arg);
    return Init_ContactThresholds_torque_mode(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::ContactThresholds msg_;
};

class Init_ContactThresholds_abs_mode
{
public:
  explicit Init_ContactThresholds_abs_mode(::multi_mode_control_msgs::msg::ContactThresholds & msg)
  : msg_(msg)
  {}
  Init_ContactThresholds_directional_mode abs_mode(::multi_mode_control_msgs::msg::ContactThresholds::_abs_mode_type arg)
  {
    msg_.abs_mode = std::move(arg);
    return Init_ContactThresholds_directional_mode(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::ContactThresholds msg_;
};

class Init_ContactThresholds_m_abs_max
{
public:
  explicit Init_ContactThresholds_m_abs_max(::multi_mode_control_msgs::msg::ContactThresholds & msg)
  : msg_(msg)
  {}
  Init_ContactThresholds_abs_mode m_abs_max(::multi_mode_control_msgs::msg::ContactThresholds::_m_abs_max_type arg)
  {
    msg_.m_abs_max = std::move(arg);
    return Init_ContactThresholds_abs_mode(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::ContactThresholds msg_;
};

class Init_ContactThresholds_f_abs_max
{
public:
  explicit Init_ContactThresholds_f_abs_max(::multi_mode_control_msgs::msg::ContactThresholds & msg)
  : msg_(msg)
  {}
  Init_ContactThresholds_m_abs_max f_abs_max(::multi_mode_control_msgs::msg::ContactThresholds::_f_abs_max_type arg)
  {
    msg_.f_abs_max = std::move(arg);
    return Init_ContactThresholds_m_abs_max(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::ContactThresholds msg_;
};

class Init_ContactThresholds_tau_min
{
public:
  explicit Init_ContactThresholds_tau_min(::multi_mode_control_msgs::msg::ContactThresholds & msg)
  : msg_(msg)
  {}
  Init_ContactThresholds_f_abs_max tau_min(::multi_mode_control_msgs::msg::ContactThresholds::_tau_min_type arg)
  {
    msg_.tau_min = std::move(arg);
    return Init_ContactThresholds_f_abs_max(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::ContactThresholds msg_;
};

class Init_ContactThresholds_tau_max
{
public:
  explicit Init_ContactThresholds_tau_max(::multi_mode_control_msgs::msg::ContactThresholds & msg)
  : msg_(msg)
  {}
  Init_ContactThresholds_tau_min tau_max(::multi_mode_control_msgs::msg::ContactThresholds::_tau_max_type arg)
  {
    msg_.tau_max = std::move(arg);
    return Init_ContactThresholds_tau_min(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::ContactThresholds msg_;
};

class Init_ContactThresholds_f_min
{
public:
  explicit Init_ContactThresholds_f_min(::multi_mode_control_msgs::msg::ContactThresholds & msg)
  : msg_(msg)
  {}
  Init_ContactThresholds_tau_max f_min(::multi_mode_control_msgs::msg::ContactThresholds::_f_min_type arg)
  {
    msg_.f_min = std::move(arg);
    return Init_ContactThresholds_tau_max(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::ContactThresholds msg_;
};

class Init_ContactThresholds_f_max
{
public:
  Init_ContactThresholds_f_max()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ContactThresholds_f_min f_max(::multi_mode_control_msgs::msg::ContactThresholds::_f_max_type arg)
  {
    msg_.f_max = std::move(arg);
    return Init_ContactThresholds_f_min(msg_);
  }

private:
  ::multi_mode_control_msgs::msg::ContactThresholds msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::msg::ContactThresholds>()
{
  return multi_mode_control_msgs::msg::builder::Init_ContactThresholds_f_max();
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__BUILDER_HPP_
