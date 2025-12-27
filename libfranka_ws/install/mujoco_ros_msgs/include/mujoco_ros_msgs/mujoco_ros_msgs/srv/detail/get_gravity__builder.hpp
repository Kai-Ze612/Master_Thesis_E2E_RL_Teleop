// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/GetGravity.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GRAVITY__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GRAVITY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/get_gravity__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_GetGravity_Request_admin_hash
{
public:
  Init_GetGravity_Request_admin_hash()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::mujoco_ros_msgs::srv::GetGravity_Request admin_hash(::mujoco_ros_msgs::srv::GetGravity_Request::_admin_hash_type arg)
  {
    msg_.admin_hash = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetGravity_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::GetGravity_Request>()
{
  return mujoco_ros_msgs::srv::builder::Init_GetGravity_Request_admin_hash();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_GetGravity_Response_status_message
{
public:
  explicit Init_GetGravity_Response_status_message(::mujoco_ros_msgs::srv::GetGravity_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::GetGravity_Response status_message(::mujoco_ros_msgs::srv::GetGravity_Response::_status_message_type arg)
  {
    msg_.status_message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetGravity_Response msg_;
};

class Init_GetGravity_Response_success
{
public:
  explicit Init_GetGravity_Response_success(::mujoco_ros_msgs::srv::GetGravity_Response & msg)
  : msg_(msg)
  {}
  Init_GetGravity_Response_status_message success(::mujoco_ros_msgs::srv::GetGravity_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GetGravity_Response_status_message(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetGravity_Response msg_;
};

class Init_GetGravity_Response_gravity
{
public:
  Init_GetGravity_Response_gravity()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetGravity_Response_success gravity(::mujoco_ros_msgs::srv::GetGravity_Response::_gravity_type arg)
  {
    msg_.gravity = std::move(arg);
    return Init_GetGravity_Response_success(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetGravity_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::GetGravity_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_GetGravity_Response_gravity();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GRAVITY__BUILDER_HPP_
