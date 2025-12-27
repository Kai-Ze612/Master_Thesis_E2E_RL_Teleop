// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/GeomProperties.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/geom_properties__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_GeomProperties_friction_roll
{
public:
  explicit Init_GeomProperties_friction_roll(::mujoco_ros_msgs::msg::GeomProperties & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::GeomProperties friction_roll(::mujoco_ros_msgs::msg::GeomProperties::_friction_roll_type arg)
  {
    msg_.friction_roll = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::GeomProperties msg_;
};

class Init_GeomProperties_friction_spin
{
public:
  explicit Init_GeomProperties_friction_spin(::mujoco_ros_msgs::msg::GeomProperties & msg)
  : msg_(msg)
  {}
  Init_GeomProperties_friction_roll friction_spin(::mujoco_ros_msgs::msg::GeomProperties::_friction_spin_type arg)
  {
    msg_.friction_spin = std::move(arg);
    return Init_GeomProperties_friction_roll(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::GeomProperties msg_;
};

class Init_GeomProperties_friction_slide
{
public:
  explicit Init_GeomProperties_friction_slide(::mujoco_ros_msgs::msg::GeomProperties & msg)
  : msg_(msg)
  {}
  Init_GeomProperties_friction_spin friction_slide(::mujoco_ros_msgs::msg::GeomProperties::_friction_slide_type arg)
  {
    msg_.friction_slide = std::move(arg);
    return Init_GeomProperties_friction_spin(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::GeomProperties msg_;
};

class Init_GeomProperties_size_2
{
public:
  explicit Init_GeomProperties_size_2(::mujoco_ros_msgs::msg::GeomProperties & msg)
  : msg_(msg)
  {}
  Init_GeomProperties_friction_slide size_2(::mujoco_ros_msgs::msg::GeomProperties::_size_2_type arg)
  {
    msg_.size_2 = std::move(arg);
    return Init_GeomProperties_friction_slide(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::GeomProperties msg_;
};

class Init_GeomProperties_size_1
{
public:
  explicit Init_GeomProperties_size_1(::mujoco_ros_msgs::msg::GeomProperties & msg)
  : msg_(msg)
  {}
  Init_GeomProperties_size_2 size_1(::mujoco_ros_msgs::msg::GeomProperties::_size_1_type arg)
  {
    msg_.size_1 = std::move(arg);
    return Init_GeomProperties_size_2(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::GeomProperties msg_;
};

class Init_GeomProperties_size_0
{
public:
  explicit Init_GeomProperties_size_0(::mujoco_ros_msgs::msg::GeomProperties & msg)
  : msg_(msg)
  {}
  Init_GeomProperties_size_1 size_0(::mujoco_ros_msgs::msg::GeomProperties::_size_0_type arg)
  {
    msg_.size_0 = std::move(arg);
    return Init_GeomProperties_size_1(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::GeomProperties msg_;
};

class Init_GeomProperties_body_mass
{
public:
  explicit Init_GeomProperties_body_mass(::mujoco_ros_msgs::msg::GeomProperties & msg)
  : msg_(msg)
  {}
  Init_GeomProperties_size_0 body_mass(::mujoco_ros_msgs::msg::GeomProperties::_body_mass_type arg)
  {
    msg_.body_mass = std::move(arg);
    return Init_GeomProperties_size_0(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::GeomProperties msg_;
};

class Init_GeomProperties_type
{
public:
  explicit Init_GeomProperties_type(::mujoco_ros_msgs::msg::GeomProperties & msg)
  : msg_(msg)
  {}
  Init_GeomProperties_body_mass type(::mujoco_ros_msgs::msg::GeomProperties::_type_type arg)
  {
    msg_.type = std::move(arg);
    return Init_GeomProperties_body_mass(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::GeomProperties msg_;
};

class Init_GeomProperties_name
{
public:
  Init_GeomProperties_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GeomProperties_type name(::mujoco_ros_msgs::msg::GeomProperties::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_GeomProperties_type(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::GeomProperties msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::GeomProperties>()
{
  return mujoco_ros_msgs::msg::builder::Init_GeomProperties_name();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__BUILDER_HPP_
