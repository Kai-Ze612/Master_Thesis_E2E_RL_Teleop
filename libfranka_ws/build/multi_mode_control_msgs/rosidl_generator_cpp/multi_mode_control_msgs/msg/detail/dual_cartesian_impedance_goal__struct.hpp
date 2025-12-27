// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from multi_mode_control_msgs:msg/DualCartesianImpedanceGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__STRUCT_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'l_pose'
// Member 'r_pose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__msg__DualCartesianImpedanceGoal __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__msg__DualCartesianImpedanceGoal __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DualCartesianImpedanceGoal_
{
  using Type = DualCartesianImpedanceGoal_<ContainerAllocator>;

  explicit DualCartesianImpedanceGoal_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : l_pose(_init),
    r_pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 7>::iterator, double>(this->l_q_n.begin(), this->l_q_n.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->r_q_n.begin(), this->r_q_n.end(), 0.0);
    }
  }

  explicit DualCartesianImpedanceGoal_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : l_pose(_alloc, _init),
    r_pose(_alloc, _init),
    l_q_n(_alloc),
    r_q_n(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 7>::iterator, double>(this->l_q_n.begin(), this->l_q_n.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->r_q_n.begin(), this->r_q_n.end(), 0.0);
    }
  }

  // field types and members
  using _l_pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _l_pose_type l_pose;
  using _r_pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _r_pose_type r_pose;
  using _l_q_n_type =
    std::array<double, 7>;
  _l_q_n_type l_q_n;
  using _r_q_n_type =
    std::array<double, 7>;
  _r_q_n_type r_q_n;

  // setters for named parameter idiom
  Type & set__l_pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->l_pose = _arg;
    return *this;
  }
  Type & set__r_pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->r_pose = _arg;
    return *this;
  }
  Type & set__l_q_n(
    const std::array<double, 7> & _arg)
  {
    this->l_q_n = _arg;
    return *this;
  }
  Type & set__r_q_n(
    const std::array<double, 7> & _arg)
  {
    this->r_q_n = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__msg__DualCartesianImpedanceGoal
    std::shared_ptr<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__msg__DualCartesianImpedanceGoal
    std::shared_ptr<multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DualCartesianImpedanceGoal_ & other) const
  {
    if (this->l_pose != other.l_pose) {
      return false;
    }
    if (this->r_pose != other.r_pose) {
      return false;
    }
    if (this->l_q_n != other.l_q_n) {
      return false;
    }
    if (this->r_q_n != other.r_q_n) {
      return false;
    }
    return true;
  }
  bool operator!=(const DualCartesianImpedanceGoal_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DualCartesianImpedanceGoal_

// alias to use template instance with default allocator
using DualCartesianImpedanceGoal =
  multi_mode_control_msgs::msg::DualCartesianImpedanceGoal_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__DUAL_CARTESIAN_IMPEDANCE_GOAL__STRUCT_HPP_
