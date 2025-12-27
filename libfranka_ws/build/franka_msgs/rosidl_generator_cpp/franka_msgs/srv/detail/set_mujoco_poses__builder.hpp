// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from franka_msgs:srv/SetMujocoPoses.idl
// generated code does not contain a copyright notice

#ifndef FRANKA_MSGS__SRV__DETAIL__SET_MUJOCO_POSES__BUILDER_HPP_
#define FRANKA_MSGS__SRV__DETAIL__SET_MUJOCO_POSES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "franka_msgs/srv/detail/set_mujoco_poses__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace franka_msgs
{

namespace srv
{

namespace builder
{

class Init_SetMujocoPoses_Request_objects
{
public:
  Init_SetMujocoPoses_Request_objects()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::franka_msgs::srv::SetMujocoPoses_Request objects(::franka_msgs::srv::SetMujocoPoses_Request::_objects_type arg)
  {
    msg_.objects = std::move(arg);
    return std::move(msg_);
  }

private:
  ::franka_msgs::srv::SetMujocoPoses_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::franka_msgs::srv::SetMujocoPoses_Request>()
{
  return franka_msgs::srv::builder::Init_SetMujocoPoses_Request_objects();
}

}  // namespace franka_msgs


namespace franka_msgs
{

namespace srv
{

namespace builder
{

class Init_SetMujocoPoses_Response_valid
{
public:
  Init_SetMujocoPoses_Response_valid()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::franka_msgs::srv::SetMujocoPoses_Response valid(::franka_msgs::srv::SetMujocoPoses_Response::_valid_type arg)
  {
    msg_.valid = std::move(arg);
    return std::move(msg_);
  }

private:
  ::franka_msgs::srv::SetMujocoPoses_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::franka_msgs::srv::SetMujocoPoses_Response>()
{
  return franka_msgs::srv::builder::Init_SetMujocoPoses_Response_valid();
}

}  // namespace franka_msgs

#endif  // FRANKA_MSGS__SRV__DETAIL__SET_MUJOCO_POSES__BUILDER_HPP_
