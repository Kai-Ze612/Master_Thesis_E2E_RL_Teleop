// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/PluginStats.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/plugin_stats__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_PluginStats_ema_steptime_last_stage
{
public:
  explicit Init_PluginStats_ema_steptime_last_stage(::mujoco_ros_msgs::msg::PluginStats & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::PluginStats ema_steptime_last_stage(::mujoco_ros_msgs::msg::PluginStats::_ema_steptime_last_stage_type arg)
  {
    msg_.ema_steptime_last_stage = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::PluginStats msg_;
};

class Init_PluginStats_ema_steptime_render
{
public:
  explicit Init_PluginStats_ema_steptime_render(::mujoco_ros_msgs::msg::PluginStats & msg)
  : msg_(msg)
  {}
  Init_PluginStats_ema_steptime_last_stage ema_steptime_render(::mujoco_ros_msgs::msg::PluginStats::_ema_steptime_render_type arg)
  {
    msg_.ema_steptime_render = std::move(arg);
    return Init_PluginStats_ema_steptime_last_stage(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::PluginStats msg_;
};

class Init_PluginStats_ema_steptime_passive
{
public:
  explicit Init_PluginStats_ema_steptime_passive(::mujoco_ros_msgs::msg::PluginStats & msg)
  : msg_(msg)
  {}
  Init_PluginStats_ema_steptime_render ema_steptime_passive(::mujoco_ros_msgs::msg::PluginStats::_ema_steptime_passive_type arg)
  {
    msg_.ema_steptime_passive = std::move(arg);
    return Init_PluginStats_ema_steptime_render(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::PluginStats msg_;
};

class Init_PluginStats_ema_steptime_control
{
public:
  explicit Init_PluginStats_ema_steptime_control(::mujoco_ros_msgs::msg::PluginStats & msg)
  : msg_(msg)
  {}
  Init_PluginStats_ema_steptime_passive ema_steptime_control(::mujoco_ros_msgs::msg::PluginStats::_ema_steptime_control_type arg)
  {
    msg_.ema_steptime_control = std::move(arg);
    return Init_PluginStats_ema_steptime_passive(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::PluginStats msg_;
};

class Init_PluginStats_reset_time
{
public:
  explicit Init_PluginStats_reset_time(::mujoco_ros_msgs::msg::PluginStats & msg)
  : msg_(msg)
  {}
  Init_PluginStats_ema_steptime_control reset_time(::mujoco_ros_msgs::msg::PluginStats::_reset_time_type arg)
  {
    msg_.reset_time = std::move(arg);
    return Init_PluginStats_ema_steptime_control(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::PluginStats msg_;
};

class Init_PluginStats_load_time
{
public:
  explicit Init_PluginStats_load_time(::mujoco_ros_msgs::msg::PluginStats & msg)
  : msg_(msg)
  {}
  Init_PluginStats_reset_time load_time(::mujoco_ros_msgs::msg::PluginStats::_load_time_type arg)
  {
    msg_.load_time = std::move(arg);
    return Init_PluginStats_reset_time(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::PluginStats msg_;
};

class Init_PluginStats_plugin_type
{
public:
  Init_PluginStats_plugin_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PluginStats_load_time plugin_type(::mujoco_ros_msgs::msg::PluginStats::_plugin_type_type arg)
  {
    msg_.plugin_type = std::move(arg);
    return Init_PluginStats_load_time(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::PluginStats msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::PluginStats>()
{
  return mujoco_ros_msgs::msg::builder::Init_PluginStats_plugin_type();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__BUILDER_HPP_
