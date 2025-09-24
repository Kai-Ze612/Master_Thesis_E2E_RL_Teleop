// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/BodyState.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__BODY_STATE__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__BODY_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/body_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_BodyState_mass
{
public:
  explicit Init_BodyState_mass(::mujoco_ros_msgs::msg::BodyState & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::BodyState mass(::mujoco_ros_msgs::msg::BodyState::_mass_type arg)
  {
    msg_.mass = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::BodyState msg_;
};

class Init_BodyState_twist
{
public:
  explicit Init_BodyState_twist(::mujoco_ros_msgs::msg::BodyState & msg)
  : msg_(msg)
  {}
  Init_BodyState_mass twist(::mujoco_ros_msgs::msg::BodyState::_twist_type arg)
  {
    msg_.twist = std::move(arg);
    return Init_BodyState_mass(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::BodyState msg_;
};

class Init_BodyState_pose
{
public:
  explicit Init_BodyState_pose(::mujoco_ros_msgs::msg::BodyState & msg)
  : msg_(msg)
  {}
  Init_BodyState_twist pose(::mujoco_ros_msgs::msg::BodyState::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return Init_BodyState_twist(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::BodyState msg_;
};

class Init_BodyState_name
{
public:
  Init_BodyState_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_BodyState_pose name(::mujoco_ros_msgs::msg::BodyState::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_BodyState_pose(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::BodyState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::BodyState>()
{
  return mujoco_ros_msgs::msg::builder::Init_BodyState_name();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__BODY_STATE__BUILDER_HPP_
