// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/SetBodyState.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__SET_BODY_STATE__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__SET_BODY_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/set_body_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_SetBodyState_Request_admin_hash
{
public:
  explicit Init_SetBodyState_Request_admin_hash(::mujoco_ros_msgs::srv::SetBodyState_Request & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::SetBodyState_Request admin_hash(::mujoco_ros_msgs::srv::SetBodyState_Request::_admin_hash_type arg)
  {
    msg_.admin_hash = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetBodyState_Request msg_;
};

class Init_SetBodyState_Request_reset_qpos
{
public:
  explicit Init_SetBodyState_Request_reset_qpos(::mujoco_ros_msgs::srv::SetBodyState_Request & msg)
  : msg_(msg)
  {}
  Init_SetBodyState_Request_admin_hash reset_qpos(::mujoco_ros_msgs::srv::SetBodyState_Request::_reset_qpos_type arg)
  {
    msg_.reset_qpos = std::move(arg);
    return Init_SetBodyState_Request_admin_hash(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetBodyState_Request msg_;
};

class Init_SetBodyState_Request_set_mass
{
public:
  explicit Init_SetBodyState_Request_set_mass(::mujoco_ros_msgs::srv::SetBodyState_Request & msg)
  : msg_(msg)
  {}
  Init_SetBodyState_Request_reset_qpos set_mass(::mujoco_ros_msgs::srv::SetBodyState_Request::_set_mass_type arg)
  {
    msg_.set_mass = std::move(arg);
    return Init_SetBodyState_Request_reset_qpos(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetBodyState_Request msg_;
};

class Init_SetBodyState_Request_set_twist
{
public:
  explicit Init_SetBodyState_Request_set_twist(::mujoco_ros_msgs::srv::SetBodyState_Request & msg)
  : msg_(msg)
  {}
  Init_SetBodyState_Request_set_mass set_twist(::mujoco_ros_msgs::srv::SetBodyState_Request::_set_twist_type arg)
  {
    msg_.set_twist = std::move(arg);
    return Init_SetBodyState_Request_set_mass(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetBodyState_Request msg_;
};

class Init_SetBodyState_Request_set_pose
{
public:
  explicit Init_SetBodyState_Request_set_pose(::mujoco_ros_msgs::srv::SetBodyState_Request & msg)
  : msg_(msg)
  {}
  Init_SetBodyState_Request_set_twist set_pose(::mujoco_ros_msgs::srv::SetBodyState_Request::_set_pose_type arg)
  {
    msg_.set_pose = std::move(arg);
    return Init_SetBodyState_Request_set_twist(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetBodyState_Request msg_;
};

class Init_SetBodyState_Request_state
{
public:
  Init_SetBodyState_Request_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetBodyState_Request_set_pose state(::mujoco_ros_msgs::srv::SetBodyState_Request::_state_type arg)
  {
    msg_.state = std::move(arg);
    return Init_SetBodyState_Request_set_pose(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetBodyState_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::SetBodyState_Request>()
{
  return mujoco_ros_msgs::srv::builder::Init_SetBodyState_Request_state();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_SetBodyState_Response_status_message
{
public:
  explicit Init_SetBodyState_Response_status_message(::mujoco_ros_msgs::srv::SetBodyState_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::SetBodyState_Response status_message(::mujoco_ros_msgs::srv::SetBodyState_Response::_status_message_type arg)
  {
    msg_.status_message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetBodyState_Response msg_;
};

class Init_SetBodyState_Response_success
{
public:
  Init_SetBodyState_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetBodyState_Response_status_message success(::mujoco_ros_msgs::srv::SetBodyState_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SetBodyState_Response_status_message(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetBodyState_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::SetBodyState_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_SetBodyState_Response_success();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__SET_BODY_STATE__BUILDER_HPP_
