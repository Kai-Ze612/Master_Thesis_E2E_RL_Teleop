// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/SetGravity.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__SET_GRAVITY__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__SET_GRAVITY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/set_gravity__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_SetGravity_Request_gravity
{
public:
  explicit Init_SetGravity_Request_gravity(::mujoco_ros_msgs::srv::SetGravity_Request & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::SetGravity_Request gravity(::mujoco_ros_msgs::srv::SetGravity_Request::_gravity_type arg)
  {
    msg_.gravity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGravity_Request msg_;
};

class Init_SetGravity_Request_admin_hash
{
public:
  Init_SetGravity_Request_admin_hash()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetGravity_Request_gravity admin_hash(::mujoco_ros_msgs::srv::SetGravity_Request::_admin_hash_type arg)
  {
    msg_.admin_hash = std::move(arg);
    return Init_SetGravity_Request_gravity(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGravity_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::SetGravity_Request>()
{
  return mujoco_ros_msgs::srv::builder::Init_SetGravity_Request_admin_hash();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_SetGravity_Response_status_message
{
public:
  explicit Init_SetGravity_Response_status_message(::mujoco_ros_msgs::srv::SetGravity_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::SetGravity_Response status_message(::mujoco_ros_msgs::srv::SetGravity_Response::_status_message_type arg)
  {
    msg_.status_message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGravity_Response msg_;
};

class Init_SetGravity_Response_success
{
public:
  Init_SetGravity_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetGravity_Response_status_message success(::mujoco_ros_msgs::srv::SetGravity_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SetGravity_Response_status_message(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGravity_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::SetGravity_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_SetGravity_Response_success();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__SET_GRAVITY__BUILDER_HPP_
