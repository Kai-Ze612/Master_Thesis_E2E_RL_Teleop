// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/GetPluginStats.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_PLUGIN_STATS__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_PLUGIN_STATS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/get_plugin_stats__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::GetPluginStats_Request>()
{
  return ::mujoco_ros_msgs::srv::GetPluginStats_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_GetPluginStats_Response_stats
{
public:
  Init_GetPluginStats_Response_stats()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::mujoco_ros_msgs::srv::GetPluginStats_Response stats(::mujoco_ros_msgs::srv::GetPluginStats_Response::_stats_type arg)
  {
    msg_.stats = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetPluginStats_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::GetPluginStats_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_GetPluginStats_Response_stats();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_PLUGIN_STATS__BUILDER_HPP_
