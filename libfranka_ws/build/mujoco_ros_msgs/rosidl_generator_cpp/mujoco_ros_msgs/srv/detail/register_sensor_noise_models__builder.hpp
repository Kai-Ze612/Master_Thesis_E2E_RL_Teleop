// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:srv/RegisterSensorNoiseModels.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__REGISTER_SENSOR_NOISE_MODELS__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__REGISTER_SENSOR_NOISE_MODELS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/srv/detail/register_sensor_noise_models__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_RegisterSensorNoiseModels_Request_admin_hash
{
public:
  explicit Init_RegisterSensorNoiseModels_Request_admin_hash(::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request admin_hash(::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request::_admin_hash_type arg)
  {
    msg_.admin_hash = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request msg_;
};

class Init_RegisterSensorNoiseModels_Request_noise_models
{
public:
  Init_RegisterSensorNoiseModels_Request_noise_models()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RegisterSensorNoiseModels_Request_admin_hash noise_models(::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request::_noise_models_type arg)
  {
    msg_.noise_models = std::move(arg);
    return Init_RegisterSensorNoiseModels_Request_admin_hash(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request>()
{
  return mujoco_ros_msgs::srv::builder::Init_RegisterSensorNoiseModels_Request_noise_models();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace srv
{

namespace builder
{

class Init_RegisterSensorNoiseModels_Response_success
{
public:
  Init_RegisterSensorNoiseModels_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response success(::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response>()
{
  return mujoco_ros_msgs::srv::builder::Init_RegisterSensorNoiseModels_Response_success();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__REGISTER_SENSOR_NOISE_MODELS__BUILDER_HPP_
