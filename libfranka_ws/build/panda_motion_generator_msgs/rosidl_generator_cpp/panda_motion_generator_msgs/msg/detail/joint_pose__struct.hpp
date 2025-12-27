// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from panda_motion_generator_msgs:msg/JointPose.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__STRUCT_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__msg__JointPose __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__msg__JointPose __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct JointPose_
{
  using Type = JointPose_<ContainerAllocator>;

  explicit JointPose_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit JointPose_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _q_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _q_type q;

  // setters for named parameter idiom
  Type & set__q(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->q = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__msg__JointPose
    std::shared_ptr<panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__msg__JointPose
    std::shared_ptr<panda_motion_generator_msgs::msg::JointPose_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JointPose_ & other) const
  {
    if (this->q != other.q) {
      return false;
    }
    return true;
  }
  bool operator!=(const JointPose_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JointPose_

// alias to use template instance with default allocator
using JointPose =
  panda_motion_generator_msgs::msg::JointPose_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace panda_motion_generator_msgs

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__JOINT_POSE__STRUCT_HPP_
