// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/GetBodyState.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_BODY_STATE__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_BODY_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/get_body_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_GetBodyState_Request_admin_hash
{
public:
  explicit Init_GetBodyState_Request_admin_hash(::mujoco_ros_msgs::srv::GetBodyState_Request & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::GetBodyState_Request admin_hash(::mujoco_ros_msgs::srv::GetBodyState_Request::_admin_hash_type arg)
  {
    msg_.admin_hash = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetBodyState_Request msg_;
};

class Init_GetBodyState_Request_name
{
public:
  Init_GetBodyState_Request_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetBodyState_Request_admin_hash name(::mujoco_ros_msgs::srv::GetBodyState_Request::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_GetBodyState_Request_admin_hash(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetBodyState_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::GetBodyState_Request>()
{
  return mujoco_ros_msgs::srv::builder::Init_GetBodyState_Request_name();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_GetBodyState_Response_status_message
{
public:
  explicit Init_GetBodyState_Response_status_message(::mujoco_ros_msgs::srv::GetBodyState_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::GetBodyState_Response status_message(::mujoco_ros_msgs::srv::GetBodyState_Response::_status_message_type arg)
  {
    msg_.status_message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetBodyState_Response msg_;
};

class Init_GetBodyState_Response_success
{
public:
  explicit Init_GetBodyState_Response_success(::mujoco_ros_msgs::srv::GetBodyState_Response & msg)
  : msg_(msg)
  {}
  Init_GetBodyState_Response_status_message success(::mujoco_ros_msgs::srv::GetBodyState_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GetBodyState_Response_status_message(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetBodyState_Response msg_;
};

class Init_GetBodyState_Response_state
{
public:
  Init_GetBodyState_Response_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetBodyState_Response_success state(::mujoco_ros_msgs::srv::GetBodyState_Response::_state_type arg)
  {
    msg_.state = std::move(arg);
    return Init_GetBodyState_Response_success(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetBodyState_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::GetBodyState_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_GetBodyState_Response_state();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_BODY_STATE__BUILDER_HPP_
