// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/SimInfo.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/sim_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_SimInfo_rt_setting
{
public:
  explicit Init_SimInfo_rt_setting(::mujoco_ros_msgs::msg::SimInfo & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::SimInfo rt_setting(::mujoco_ros_msgs::msg::SimInfo::_rt_setting_type arg)
  {
    msg_.rt_setting = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SimInfo msg_;
};

class Init_SimInfo_rt_measured
{
public:
  explicit Init_SimInfo_rt_measured(::mujoco_ros_msgs::msg::SimInfo & msg)
  : msg_(msg)
  {}
  Init_SimInfo_rt_setting rt_measured(::mujoco_ros_msgs::msg::SimInfo::_rt_measured_type arg)
  {
    msg_.rt_measured = std::move(arg);
    return Init_SimInfo_rt_setting(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SimInfo msg_;
};

class Init_SimInfo_pending_sim_steps
{
public:
  explicit Init_SimInfo_pending_sim_steps(::mujoco_ros_msgs::msg::SimInfo & msg)
  : msg_(msg)
  {}
  Init_SimInfo_rt_measured pending_sim_steps(::mujoco_ros_msgs::msg::SimInfo::_pending_sim_steps_type arg)
  {
    msg_.pending_sim_steps = std::move(arg);
    return Init_SimInfo_rt_measured(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SimInfo msg_;
};

class Init_SimInfo_paused
{
public:
  explicit Init_SimInfo_paused(::mujoco_ros_msgs::msg::SimInfo & msg)
  : msg_(msg)
  {}
  Init_SimInfo_pending_sim_steps paused(::mujoco_ros_msgs::msg::SimInfo::_paused_type arg)
  {
    msg_.paused = std::move(arg);
    return Init_SimInfo_pending_sim_steps(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SimInfo msg_;
};

class Init_SimInfo_loading_state
{
public:
  explicit Init_SimInfo_loading_state(::mujoco_ros_msgs::msg::SimInfo & msg)
  : msg_(msg)
  {}
  Init_SimInfo_paused loading_state(::mujoco_ros_msgs::msg::SimInfo::_loading_state_type arg)
  {
    msg_.loading_state = std::move(arg);
    return Init_SimInfo_paused(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SimInfo msg_;
};

class Init_SimInfo_load_count
{
public:
  explicit Init_SimInfo_load_count(::mujoco_ros_msgs::msg::SimInfo & msg)
  : msg_(msg)
  {}
  Init_SimInfo_loading_state load_count(::mujoco_ros_msgs::msg::SimInfo::_load_count_type arg)
  {
    msg_.load_count = std::move(arg);
    return Init_SimInfo_loading_state(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SimInfo msg_;
};

class Init_SimInfo_model_valid
{
public:
  explicit Init_SimInfo_model_valid(::mujoco_ros_msgs::msg::SimInfo & msg)
  : msg_(msg)
  {}
  Init_SimInfo_load_count model_valid(::mujoco_ros_msgs::msg::SimInfo::_model_valid_type arg)
  {
    msg_.model_valid = std::move(arg);
    return Init_SimInfo_load_count(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SimInfo msg_;
};

class Init_SimInfo_model_path
{
public:
  Init_SimInfo_model_path()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SimInfo_model_valid model_path(::mujoco_ros_msgs::msg::SimInfo::_model_path_type arg)
  {
    msg_.model_path = std::move(arg);
    return Init_SimInfo_model_valid(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SimInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::SimInfo>()
{
  return mujoco_ros_msgs::msg::builder::Init_SimInfo_model_path();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__BUILDER_HPP_
