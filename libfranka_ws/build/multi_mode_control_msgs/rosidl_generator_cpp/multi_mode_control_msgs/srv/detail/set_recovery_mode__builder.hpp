// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:srv/SetRecoveryMode.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_RECOVERY_MODE__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_RECOVERY_MODE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/srv/detail/set_recovery_mode__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace srv
{

namespace builder
{

class Init_SetRecoveryMode_Request_timeout
{
public:
  explicit Init_SetRecoveryMode_Request_timeout(::multi_mode_control_msgs::srv::SetRecoveryMode_Request & msg)
  : msg_(msg)
  {}
  ::multi_mode_control_msgs::srv::SetRecoveryMode_Request timeout(::multi_mode_control_msgs::srv::SetRecoveryMode_Request::_timeout_type arg)
  {
    msg_.timeout = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::srv::SetRecoveryMode_Request msg_;
};

class Init_SetRecoveryMode_Request_mode
{
public:
  Init_SetRecoveryMode_Request_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetRecoveryMode_Request_timeout mode(::multi_mode_control_msgs::srv::SetRecoveryMode_Request::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_SetRecoveryMode_Request_timeout(msg_);
  }

private:
  ::multi_mode_control_msgs::srv::SetRecoveryMode_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::srv::SetRecoveryMode_Request>()
{
  return multi_mode_control_msgs::srv::builder::Init_SetRecoveryMode_Request_mode();
}

}  // namespace multi_mode_control_msgs


namespace multi_mode_control_msgs
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::srv::SetRecoveryMode_Response>()
{
  return ::multi_mode_control_msgs::srv::SetRecoveryMode_Response(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_RECOVERY_MODE__BUILDER_HPP_
