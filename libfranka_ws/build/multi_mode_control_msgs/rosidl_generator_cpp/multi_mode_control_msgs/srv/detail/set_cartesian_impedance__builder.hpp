// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from multi_mode_control_msgs:srv/SetCartesianImpedance.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_CARTESIAN_IMPEDANCE__BUILDER_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_CARTESIAN_IMPEDANCE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "multi_mode_control_msgs/srv/detail/set_cartesian_impedance__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace multi_mode_control_msgs
{

namespace srv
{

namespace builder
{

class Init_SetCartesianImpedance_Request_nullspace_stiffness
{
public:
  explicit Init_SetCartesianImpedance_Request_nullspace_stiffness(::multi_mode_control_msgs::srv::SetCartesianImpedance_Request & msg)
  : msg_(msg)
  {}
  ::multi_mode_control_msgs::srv::SetCartesianImpedance_Request nullspace_stiffness(::multi_mode_control_msgs::srv::SetCartesianImpedance_Request::_nullspace_stiffness_type arg)
  {
    msg_.nullspace_stiffness = std::move(arg);
    return std::move(msg_);
  }

private:
  ::multi_mode_control_msgs::srv::SetCartesianImpedance_Request msg_;
};

class Init_SetCartesianImpedance_Request_damping_ratio
{
public:
  explicit Init_SetCartesianImpedance_Request_damping_ratio(::multi_mode_control_msgs::srv::SetCartesianImpedance_Request & msg)
  : msg_(msg)
  {}
  Init_SetCartesianImpedance_Request_nullspace_stiffness damping_ratio(::multi_mode_control_msgs::srv::SetCartesianImpedance_Request::_damping_ratio_type arg)
  {
    msg_.damping_ratio = std::move(arg);
    return Init_SetCartesianImpedance_Request_nullspace_stiffness(msg_);
  }

private:
  ::multi_mode_control_msgs::srv::SetCartesianImpedance_Request msg_;
};

class Init_SetCartesianImpedance_Request_stiffness
{
public:
  Init_SetCartesianImpedance_Request_stiffness()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetCartesianImpedance_Request_damping_ratio stiffness(::multi_mode_control_msgs::srv::SetCartesianImpedance_Request::_stiffness_type arg)
  {
    msg_.stiffness = std::move(arg);
    return Init_SetCartesianImpedance_Request_damping_ratio(msg_);
  }

private:
  ::multi_mode_control_msgs::srv::SetCartesianImpedance_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::srv::SetCartesianImpedance_Request>()
{
  return multi_mode_control_msgs::srv::builder::Init_SetCartesianImpedance_Request_stiffness();
}

}  // namespace multi_mode_control_msgs


namespace multi_mode_control_msgs
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::multi_mode_control_msgs::srv::SetCartesianImpedance_Response>()
{
  return ::multi_mode_control_msgs::srv::SetCartesianImpedance_Response(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_CARTESIAN_IMPEDANCE__BUILDER_HPP_
