// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/ResetBodyQPos.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__RESET_BODY_Q_POS__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__RESET_BODY_Q_POS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/reset_body_q_pos__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_ResetBodyQPos_Request_admin_hash
{
public:
  explicit Init_ResetBodyQPos_Request_admin_hash(::mujoco_ros_msgs::srv::ResetBodyQPos_Request & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::ResetBodyQPos_Request admin_hash(::mujoco_ros_msgs::srv::ResetBodyQPos_Request::_admin_hash_type arg)
  {
    msg_.admin_hash = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::ResetBodyQPos_Request msg_;
};

class Init_ResetBodyQPos_Request_name
{
public:
  Init_ResetBodyQPos_Request_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ResetBodyQPos_Request_admin_hash name(::mujoco_ros_msgs::srv::ResetBodyQPos_Request::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_ResetBodyQPos_Request_admin_hash(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::ResetBodyQPos_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::ResetBodyQPos_Request>()
{
  return mujoco_ros_msgs::srv::builder::Init_ResetBodyQPos_Request_name();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_ResetBodyQPos_Response_status_message
{
public:
  explicit Init_ResetBodyQPos_Response_status_message(::mujoco_ros_msgs::srv::ResetBodyQPos_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::ResetBodyQPos_Response status_message(::mujoco_ros_msgs::srv::ResetBodyQPos_Response::_status_message_type arg)
  {
    msg_.status_message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::ResetBodyQPos_Response msg_;
};

class Init_ResetBodyQPos_Response_success
{
public:
  Init_ResetBodyQPos_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ResetBodyQPos_Response_status_message success(::mujoco_ros_msgs::srv::ResetBodyQPos_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_ResetBodyQPos_Response_status_message(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::ResetBodyQPos_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::ResetBodyQPos_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_ResetBodyQPos_Response_success();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__RESET_BODY_Q_POS__BUILDER_HPP_
