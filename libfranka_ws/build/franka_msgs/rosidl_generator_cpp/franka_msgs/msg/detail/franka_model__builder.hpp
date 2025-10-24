// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from franka_msgs:msg/FrankaModel.idl
// generated code does not contain a copyright notice

#ifndef FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__BUILDER_HPP_
#define FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "franka_msgs/msg/detail/franka_model__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace franka_msgs
{

namespace msg
{

namespace builder
{

class Init_FrankaModel_mass
{
public:
  explicit Init_FrankaModel_mass(::franka_msgs::msg::FrankaModel & msg)
  : msg_(msg)
  {}
  ::franka_msgs::msg::FrankaModel mass(::franka_msgs::msg::FrankaModel::_mass_type arg)
  {
    msg_.mass = std::move(arg);
    return std::move(msg_);
  }

private:
  ::franka_msgs::msg::FrankaModel msg_;
};

class Init_FrankaModel_ee_body_jacobian
{
public:
  explicit Init_FrankaModel_ee_body_jacobian(::franka_msgs::msg::FrankaModel & msg)
  : msg_(msg)
  {}
  Init_FrankaModel_mass ee_body_jacobian(::franka_msgs::msg::FrankaModel::_ee_body_jacobian_type arg)
  {
    msg_.ee_body_jacobian = std::move(arg);
    return Init_FrankaModel_mass(msg_);
  }

private:
  ::franka_msgs::msg::FrankaModel msg_;
};

class Init_FrankaModel_ee_zero_jacobian
{
public:
  explicit Init_FrankaModel_ee_zero_jacobian(::franka_msgs::msg::FrankaModel & msg)
  : msg_(msg)
  {}
  Init_FrankaModel_ee_body_jacobian ee_zero_jacobian(::franka_msgs::msg::FrankaModel::_ee_zero_jacobian_type arg)
  {
    msg_.ee_zero_jacobian = std::move(arg);
    return Init_FrankaModel_ee_body_jacobian(msg_);
  }

private:
  ::franka_msgs::msg::FrankaModel msg_;
};

class Init_FrankaModel_coriolis
{
public:
  explicit Init_FrankaModel_coriolis(::franka_msgs::msg::FrankaModel & msg)
  : msg_(msg)
  {}
  Init_FrankaModel_ee_zero_jacobian coriolis(::franka_msgs::msg::FrankaModel::_coriolis_type arg)
  {
    msg_.coriolis = std::move(arg);
    return Init_FrankaModel_ee_zero_jacobian(msg_);
  }

private:
  ::franka_msgs::msg::FrankaModel msg_;
};

class Init_FrankaModel_header
{
public:
  Init_FrankaModel_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FrankaModel_coriolis header(::franka_msgs::msg::FrankaModel::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_FrankaModel_coriolis(msg_);
  }

private:
  ::franka_msgs::msg::FrankaModel msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::franka_msgs::msg::FrankaModel>()
{
  return franka_msgs::msg::builder::Init_FrankaModel_header();
}

}  // namespace franka_msgs

#endif  // FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__BUILDER_HPP_
