// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/MocapState.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__MOCAP_STATE__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__MOCAP_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/mocap_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_MocapState_pose
{
public:
  explicit Init_MocapState_pose(::mujoco_ros_msgs::msg::MocapState & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::MocapState pose(::mujoco_ros_msgs::msg::MocapState::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::MocapState msg_;
};

class Init_MocapState_name
{
public:
  Init_MocapState_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MocapState_pose name(::mujoco_ros_msgs::msg::MocapState::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_MocapState_pose(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::MocapState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::MocapState>()
{
  return mujoco_ros_msgs::msg::builder::Init_MocapState_name();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__MOCAP_STATE__BUILDER_HPP_
