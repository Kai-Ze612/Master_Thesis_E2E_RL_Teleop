// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:msg/SolverParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/msg/detail/solver_parameters__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace msg
{

namespace builder
{

class Init_SolverParameters_dampratio
{
public:
  explicit Init_SolverParameters_dampratio(::mujoco_ros_msgs::msg::SolverParameters & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::msg::SolverParameters dampratio(::mujoco_ros_msgs::msg::SolverParameters::_dampratio_type arg)
  {
    msg_.dampratio = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SolverParameters msg_;
};

class Init_SolverParameters_timeconst
{
public:
  explicit Init_SolverParameters_timeconst(::mujoco_ros_msgs::msg::SolverParameters & msg)
  : msg_(msg)
  {}
  Init_SolverParameters_dampratio timeconst(::mujoco_ros_msgs::msg::SolverParameters::_timeconst_type arg)
  {
    msg_.timeconst = std::move(arg);
    return Init_SolverParameters_dampratio(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SolverParameters msg_;
};

class Init_SolverParameters_power
{
public:
  explicit Init_SolverParameters_power(::mujoco_ros_msgs::msg::SolverParameters & msg)
  : msg_(msg)
  {}
  Init_SolverParameters_timeconst power(::mujoco_ros_msgs::msg::SolverParameters::_power_type arg)
  {
    msg_.power = std::move(arg);
    return Init_SolverParameters_timeconst(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SolverParameters msg_;
};

class Init_SolverParameters_midpoint
{
public:
  explicit Init_SolverParameters_midpoint(::mujoco_ros_msgs::msg::SolverParameters & msg)
  : msg_(msg)
  {}
  Init_SolverParameters_power midpoint(::mujoco_ros_msgs::msg::SolverParameters::_midpoint_type arg)
  {
    msg_.midpoint = std::move(arg);
    return Init_SolverParameters_power(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SolverParameters msg_;
};

class Init_SolverParameters_width
{
public:
  explicit Init_SolverParameters_width(::mujoco_ros_msgs::msg::SolverParameters & msg)
  : msg_(msg)
  {}
  Init_SolverParameters_midpoint width(::mujoco_ros_msgs::msg::SolverParameters::_width_type arg)
  {
    msg_.width = std::move(arg);
    return Init_SolverParameters_midpoint(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SolverParameters msg_;
};

class Init_SolverParameters_dmax
{
public:
  explicit Init_SolverParameters_dmax(::mujoco_ros_msgs::msg::SolverParameters & msg)
  : msg_(msg)
  {}
  Init_SolverParameters_width dmax(::mujoco_ros_msgs::msg::SolverParameters::_dmax_type arg)
  {
    msg_.dmax = std::move(arg);
    return Init_SolverParameters_width(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SolverParameters msg_;
};

class Init_SolverParameters_dmin
{
public:
  Init_SolverParameters_dmin()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SolverParameters_dmax dmin(::mujoco_ros_msgs::msg::SolverParameters::_dmin_type arg)
  {
    msg_.dmin = std::move(arg);
    return Init_SolverParameters_dmax(msg_);
  }

private:
  ::mujoco_ros_msgs::msg::SolverParameters msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::msg::SolverParameters>()
{
  return mujoco_ros_msgs::msg::builder::Init_SolverParameters_dmin();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__BUILDER_HPP_
