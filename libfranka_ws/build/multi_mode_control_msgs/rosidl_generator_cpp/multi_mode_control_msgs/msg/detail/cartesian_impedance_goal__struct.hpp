// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from multi_mode_control_msgs:msg/CartesianImpedanceGoal.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CARTESIAN_IMPEDANCE_GOAL__STRUCT_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CARTESIAN_IMPEDANCE_GOAL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__msg__CartesianImpedanceGoal __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__msg__CartesianImpedanceGoal __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CartesianImpedanceGoal_
{
  using Type = CartesianImpedanceGoal_<ContainerAllocator>;

  explicit CartesianImpedanceGoal_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 7>::iterator, double>(this->q_n.begin(), this->q_n.end(), 0.0);
    }
  }

  explicit CartesianImpedanceGoal_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pose(_alloc, _init),
    q_n(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 7>::iterator, double>(this->q_n.begin(), this->q_n.end(), 0.0);
    }
  }

  // field types and members
  using _pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _pose_type pose;
  using _q_n_type =
    std::array<double, 7>;
  _q_n_type q_n;

  // setters for named parameter idiom
  Type & set__pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->pose = _arg;
    return *this;
  }
  Type & set__q_n(
    const std::array<double, 7> & _arg)
  {
    this->q_n = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__msg__CartesianImpedanceGoal
    std::shared_ptr<multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__msg__CartesianImpedanceGoal
    std::shared_ptr<multi_mode_control_msgs::msg::CartesianImpedanceGoal_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianImpedanceGoal_ & other) const
  {
    if (this->pose != other.pose) {
      return false;
    }
    if (this->q_n != other.q_n) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianImpedanceGoal_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianImpedanceGoal_

// alias to use template instance with default allocator
using CartesianImpedanceGoal =
  multi_mode_control_msgs::msg::CartesianImpedanceGoal_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CARTESIAN_IMPEDANCE_GOAL__STRUCT_HPP_
