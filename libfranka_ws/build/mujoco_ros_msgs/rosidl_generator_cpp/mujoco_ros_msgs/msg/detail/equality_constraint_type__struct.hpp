// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:msg/EqualityConstraintType.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_TYPE__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_TYPE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__msg__EqualityConstraintType __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__msg__EqualityConstraintType __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct EqualityConstraintType_
{
  using Type = EqualityConstraintType_<ContainerAllocator>;

  explicit EqualityConstraintType_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->value = 0;
    }
  }

  explicit EqualityConstraintType_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->value = 0;
    }
  }

  // field types and members
  using _value_type =
    uint16_t;
  _value_type value;

  // setters for named parameter idiom
  Type & set__value(
    const uint16_t & _arg)
  {
    this->value = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint16_t CONNECT =
    0u;
  static constexpr uint16_t WELD =
    1u;
  static constexpr uint16_t JOINT =
    2u;
  static constexpr uint16_t TENDON =
    3u;

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__msg__EqualityConstraintType
    std::shared_ptr<mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__msg__EqualityConstraintType
    std::shared_ptr<mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const EqualityConstraintType_ & other) const
  {
    if (this->value != other.value) {
      return false;
    }
    return true;
  }
  bool operator!=(const EqualityConstraintType_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct EqualityConstraintType_

// alias to use template instance with default allocator
using EqualityConstraintType =
  mujoco_ros_msgs::msg::EqualityConstraintType_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint16_t EqualityConstraintType_<ContainerAllocator>::CONNECT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint16_t EqualityConstraintType_<ContainerAllocator>::WELD;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint16_t EqualityConstraintType_<ContainerAllocator>::JOINT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint16_t EqualityConstraintType_<ContainerAllocator>::TENDON;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_TYPE__STRUCT_HPP_
