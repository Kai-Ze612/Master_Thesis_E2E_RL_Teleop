// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:srv/SetWorldState.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_WORLD_STATE__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_WORLD_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/srv/detail/set_world_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace srv
{

namespace builder
{

class Init_SetWorldState_Request_values
{
public:
  explicit Init_SetWorldState_Request_values(::multi_mode_control_msgs::srv::SetWorldState_Request & msg)
  : msg_(msg)
  {}
  ::multi_mode_control_msgs::srv::SetWorldState_Request values(::multi_mode_control_msgs::srv::SetWorldState_Request::_values_type arg)
  {
    msg_.values = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::srv::SetWorldState_Request msg_;
};

class Init_SetWorldState_Request_items
{
public:
  Init_SetWorldState_Request_items()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetWorldState_Request_values items(::multi_mode_control_msgs::srv::SetWorldState_Request::_items_type arg)
  {
    msg_.items = std::move(arg);
    return Init_SetWorldState_Request_values(msg_);
  }

private:
  ::multi_mode_control_msgs::srv::SetWorldState_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::srv::SetWorldState_Request>()
{
  return multi_mode_control_msgs::srv::builder::Init_SetWorldState_Request_items();
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
auto build<::multi_mode_control_msgs::srv::SetWorldState_Response>()
{
  return ::multi_mode_control_msgs::srv::SetWorldState_Response(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_WORLD_STATE__BUILDER_HPP_
