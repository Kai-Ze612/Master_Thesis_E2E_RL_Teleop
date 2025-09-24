// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/SetGeomProperties.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__SET_GEOM_PROPERTIES__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__SET_GEOM_PROPERTIES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/set_geom_properties__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_SetGeomProperties_Request_admin_hash
{
public:
  explicit Init_SetGeomProperties_Request_admin_hash(::mujoco_ros_msgs::srv::SetGeomProperties_Request & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::SetGeomProperties_Request admin_hash(::mujoco_ros_msgs::srv::SetGeomProperties_Request::_admin_hash_type arg)
  {
    msg_.admin_hash = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGeomProperties_Request msg_;
};

class Init_SetGeomProperties_Request_set_size
{
public:
  explicit Init_SetGeomProperties_Request_set_size(::mujoco_ros_msgs::srv::SetGeomProperties_Request & msg)
  : msg_(msg)
  {}
  Init_SetGeomProperties_Request_admin_hash set_size(::mujoco_ros_msgs::srv::SetGeomProperties_Request::_set_size_type arg)
  {
    msg_.set_size = std::move(arg);
    return Init_SetGeomProperties_Request_admin_hash(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGeomProperties_Request msg_;
};

class Init_SetGeomProperties_Request_set_friction
{
public:
  explicit Init_SetGeomProperties_Request_set_friction(::mujoco_ros_msgs::srv::SetGeomProperties_Request & msg)
  : msg_(msg)
  {}
  Init_SetGeomProperties_Request_set_size set_friction(::mujoco_ros_msgs::srv::SetGeomProperties_Request::_set_friction_type arg)
  {
    msg_.set_friction = std::move(arg);
    return Init_SetGeomProperties_Request_set_size(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGeomProperties_Request msg_;
};

class Init_SetGeomProperties_Request_set_mass
{
public:
  explicit Init_SetGeomProperties_Request_set_mass(::mujoco_ros_msgs::srv::SetGeomProperties_Request & msg)
  : msg_(msg)
  {}
  Init_SetGeomProperties_Request_set_friction set_mass(::mujoco_ros_msgs::srv::SetGeomProperties_Request::_set_mass_type arg)
  {
    msg_.set_mass = std::move(arg);
    return Init_SetGeomProperties_Request_set_friction(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGeomProperties_Request msg_;
};

class Init_SetGeomProperties_Request_set_type
{
public:
  explicit Init_SetGeomProperties_Request_set_type(::mujoco_ros_msgs::srv::SetGeomProperties_Request & msg)
  : msg_(msg)
  {}
  Init_SetGeomProperties_Request_set_mass set_type(::mujoco_ros_msgs::srv::SetGeomProperties_Request::_set_type_type arg)
  {
    msg_.set_type = std::move(arg);
    return Init_SetGeomProperties_Request_set_mass(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGeomProperties_Request msg_;
};

class Init_SetGeomProperties_Request_properties
{
public:
  Init_SetGeomProperties_Request_properties()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetGeomProperties_Request_set_type properties(::mujoco_ros_msgs::srv::SetGeomProperties_Request::_properties_type arg)
  {
    msg_.properties = std::move(arg);
    return Init_SetGeomProperties_Request_set_type(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGeomProperties_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::SetGeomProperties_Request>()
{
  return mujoco_ros_msgs::srv::builder::Init_SetGeomProperties_Request_properties();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_SetGeomProperties_Response_status_message
{
public:
  explicit Init_SetGeomProperties_Response_status_message(::mujoco_ros_msgs::srv::SetGeomProperties_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::SetGeomProperties_Response status_message(::mujoco_ros_msgs::srv::SetGeomProperties_Response::_status_message_type arg)
  {
    msg_.status_message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGeomProperties_Response msg_;
};

class Init_SetGeomProperties_Response_success
{
public:
  Init_SetGeomProperties_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetGeomProperties_Response_status_message success(::mujoco_ros_msgs::srv::SetGeomProperties_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SetGeomProperties_Response_status_message(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::SetGeomProperties_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::SetGeomProperties_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_SetGeomProperties_Response_success();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__SET_GEOM_PROPERTIES__BUILDER_HPP_
