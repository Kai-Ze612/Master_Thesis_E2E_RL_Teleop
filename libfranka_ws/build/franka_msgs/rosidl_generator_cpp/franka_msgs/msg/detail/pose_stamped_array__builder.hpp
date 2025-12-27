// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from franka_msgs:msg/PoseStampedArray.idl
// generated code does not contain a copyright notice

#ifndef FRANKA_MSGS__MSG__DETAIL__POSE_STAMPED_ARRAY__BUILDER_HPP_
#define FRANKA_MSGS__MSG__DETAIL__POSE_STAMPED_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "franka_msgs/msg/detail/pose_stamped_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace franka_msgs
{

namespace msg
{

namespace builder
{

class Init_PoseStampedArray_stampedposes
{
public:
  Init_PoseStampedArray_stampedposes()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::franka_msgs::msg::PoseStampedArray stampedposes(::franka_msgs::msg::PoseStampedArray::_stampedposes_type arg)
  {
    msg_.stampedposes = std::move(arg);
    return std::move(msg_);
  }

private:
  ::franka_msgs::msg::PoseStampedArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::franka_msgs::msg::PoseStampedArray>()
{
  return franka_msgs::msg::builder::Init_PoseStampedArray_stampedposes();
}

}  // namespace franka_msgs

#endif  // FRANKA_MSGS__MSG__DETAIL__POSE_STAMPED_ARRAY__BUILDER_HPP_
