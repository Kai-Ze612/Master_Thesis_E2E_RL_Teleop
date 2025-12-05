// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/GetGeomProperties.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GEOM_PROPERTIES__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GEOM_PROPERTIES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/get_geom_properties__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_GetGeomProperties_Request_admin_hash
{
public:
  explicit Init_GetGeomProperties_Request_admin_hash(::mujoco_ros_msgs::srv::GetGeomProperties_Request & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::GetGeomProperties_Request admin_hash(::mujoco_ros_msgs::srv::GetGeomProperties_Request::_admin_hash_type arg)
  {
    msg_.admin_hash = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetGeomProperties_Request msg_;
};

class Init_GetGeomProperties_Request_geom_name
{
public:
  Init_GetGeomProperties_Request_geom_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetGeomProperties_Request_admin_hash geom_name(::mujoco_ros_msgs::srv::GetGeomProperties_Request::_geom_name_type arg)
  {
    msg_.geom_name = std::move(arg);
    return Init_GetGeomProperties_Request_admin_hash(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetGeomProperties_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::GetGeomProperties_Request>()
{
  return mujoco_ros_msgs::srv::builder::Init_GetGeomProperties_Request_geom_name();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_GetGeomProperties_Response_status_message
{
public:
  explicit Init_GetGeomProperties_Response_status_message(::mujoco_ros_msgs::srv::GetGeomProperties_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::GetGeomProperties_Response status_message(::mujoco_ros_msgs::srv::GetGeomProperties_Response::_status_message_type arg)
  {
    msg_.status_message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetGeomProperties_Response msg_;
};

class Init_GetGeomProperties_Response_success
{
public:
  explicit Init_GetGeomProperties_Response_success(::mujoco_ros_msgs::srv::GetGeomProperties_Response & msg)
  : msg_(msg)
  {}
  Init_GetGeomProperties_Response_status_message success(::mujoco_ros_msgs::srv::GetGeomProperties_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GetGeomProperties_Response_status_message(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetGeomProperties_Response msg_;
};

class Init_GetGeomProperties_Response_properties
{
public:
  Init_GetGeomProperties_Response_properties()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetGeomProperties_Response_success properties(::mujoco_ros_msgs::srv::GetGeomProperties_Response::_properties_type arg)
  {
    msg_.properties = std::move(arg);
    return Init_GetGeomProperties_Response_success(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::GetGeomProperties_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::GetGeomProperties_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_GetGeomProperties_Response_properties();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GEOM_PROPERTIES__BUILDER_HPP_
