// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/Reload.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__RELOAD__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__RELOAD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/reload__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_Reload_Request_admin_hash
{
public:
  explicit Init_Reload_Request_admin_hash(::mujoco_ros_msgs::srv::Reload_Request & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::Reload_Request admin_hash(::mujoco_ros_msgs::srv::Reload_Request::_admin_hash_type arg)
  {
    msg_.admin_hash = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::Reload_Request msg_;
};

class Init_Reload_Request_model
{
public:
  Init_Reload_Request_model()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Reload_Request_admin_hash model(::mujoco_ros_msgs::srv::Reload_Request::_model_type arg)
  {
    msg_.model = std::move(arg);
    return Init_Reload_Request_admin_hash(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::Reload_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::Reload_Request>()
{
  return mujoco_ros_msgs::srv::builder::Init_Reload_Request_model();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_Reload_Response_status_message
{
public:
  explicit Init_Reload_Response_status_message(::mujoco_ros_msgs::srv::Reload_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::Reload_Response status_message(::mujoco_ros_msgs::srv::Reload_Response::_status_message_type arg)
  {
    msg_.status_message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::Reload_Response msg_;
};

class Init_Reload_Response_success
{
public:
  Init_Reload_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Reload_Response_status_message success(::mujoco_ros_msgs::srv::Reload_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_Reload_Response_status_message(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::Reload_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::Reload_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_Reload_Response_success();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__RELOAD__BUILDER_HPP_
