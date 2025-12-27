// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/EqualityConstraintParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/equality_constraint_parameters__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_EqualityConstraintParameters_polycoef
{
public:
  explicit Init_EqualityConstraintParameters_polycoef(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters polycoef(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_polycoef_type arg)
  {
    msg_.polycoef = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_relpose
{
public:
  explicit Init_EqualityConstraintParameters_relpose(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  Init_EqualityConstraintParameters_polycoef relpose(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_relpose_type arg)
  {
    msg_.relpose = std::move(arg);
    return Init_EqualityConstraintParameters_polycoef(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_anchor
{
public:
  explicit Init_EqualityConstraintParameters_anchor(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  Init_EqualityConstraintParameters_relpose anchor(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_anchor_type arg)
  {
    msg_.anchor = std::move(arg);
    return Init_EqualityConstraintParameters_relpose(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_torquescale
{
public:
  explicit Init_EqualityConstraintParameters_torquescale(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  Init_EqualityConstraintParameters_anchor torquescale(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_torquescale_type arg)
  {
    msg_.torquescale = std::move(arg);
    return Init_EqualityConstraintParameters_anchor(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_element2
{
public:
  explicit Init_EqualityConstraintParameters_element2(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  Init_EqualityConstraintParameters_torquescale element2(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_element2_type arg)
  {
    msg_.element2 = std::move(arg);
    return Init_EqualityConstraintParameters_torquescale(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_element1
{
public:
  explicit Init_EqualityConstraintParameters_element1(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  Init_EqualityConstraintParameters_element2 element1(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_element1_type arg)
  {
    msg_.element1 = std::move(arg);
    return Init_EqualityConstraintParameters_element2(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_class_param
{
public:
  explicit Init_EqualityConstraintParameters_class_param(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  Init_EqualityConstraintParameters_element1 class_param(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_class_param_type arg)
  {
    msg_.class_param = std::move(arg);
    return Init_EqualityConstraintParameters_element1(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_active
{
public:
  explicit Init_EqualityConstraintParameters_active(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  Init_EqualityConstraintParameters_class_param active(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_active_type arg)
  {
    msg_.active = std::move(arg);
    return Init_EqualityConstraintParameters_class_param(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_solver_parameters
{
public:
  explicit Init_EqualityConstraintParameters_solver_parameters(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  Init_EqualityConstraintParameters_active solver_parameters(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_solver_parameters_type arg)
  {
    msg_.solver_parameters = std::move(arg);
    return Init_EqualityConstraintParameters_active(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_type
{
public:
  explicit Init_EqualityConstraintParameters_type(::mujoco_ros_msgs::msg::EqualityConstraintParameters & msg)
  : msg_(msg)
  {}
  Init_EqualityConstraintParameters_solver_parameters type(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_type_type arg)
  {
    msg_.type = std::move(arg);
    return Init_EqualityConstraintParameters_solver_parameters(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

class Init_EqualityConstraintParameters_name
{
public:
  Init_EqualityConstraintParameters_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_EqualityConstraintParameters_type name(::mujoco_ros_msgs::msg::EqualityConstraintParameters::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_EqualityConstraintParameters_type(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::EqualityConstraintParameters msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::EqualityConstraintParameters>()
{
  return mujoco_ros_msgs::msg::builder::Init_EqualityConstraintParameters_name();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__BUILDER_HPP_
