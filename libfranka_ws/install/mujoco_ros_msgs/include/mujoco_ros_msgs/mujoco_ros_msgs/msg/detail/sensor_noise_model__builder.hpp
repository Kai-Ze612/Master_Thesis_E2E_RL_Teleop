// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/SensorNoiseModel.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/sensor_noise_model__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_SensorNoiseModel_set_flag
{
public:
  explicit Init_SensorNoiseModel_set_flag(::mujoco_ros_msgs::msg::SensorNoiseModel & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::SensorNoiseModel set_flag(::mujoco_ros_msgs::msg::SensorNoiseModel::_set_flag_type arg)
  {
    msg_.set_flag = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SensorNoiseModel msg_;
};

class Init_SensorNoiseModel_std
{
public:
  explicit Init_SensorNoiseModel_std(::mujoco_ros_msgs::msg::SensorNoiseModel & msg)
  : msg_(msg)
  {}
  Init_SensorNoiseModel_set_flag std(::mujoco_ros_msgs::msg::SensorNoiseModel::_std_type arg)
  {
    msg_.std = std::move(arg);
    return Init_SensorNoiseModel_set_flag(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SensorNoiseModel msg_;
};

class Init_SensorNoiseModel_mean
{
public:
  explicit Init_SensorNoiseModel_mean(::mujoco_ros_msgs::msg::SensorNoiseModel & msg)
  : msg_(msg)
  {}
  Init_SensorNoiseModel_std mean(::mujoco_ros_msgs::msg::SensorNoiseModel::_mean_type arg)
  {
    msg_.mean = std::move(arg);
    return Init_SensorNoiseModel_std(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SensorNoiseModel msg_;
};

class Init_SensorNoiseModel_sensor_name
{
public:
  Init_SensorNoiseModel_sensor_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SensorNoiseModel_mean sensor_name(::mujoco_ros_msgs::msg::SensorNoiseModel::_sensor_name_type arg)
  {
    msg_.sensor_name = std::move(arg);
    return Init_SensorNoiseModel_mean(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SensorNoiseModel msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::SensorNoiseModel>()
{
  return mujoco_ros_msgs::msg::builder::Init_SensorNoiseModel_sensor_name();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__BUILDER_HPP_
