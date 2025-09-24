// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from panda_motion_generator_msgs:msg/SetTimeScaling.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__STRUCT_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__msg__SetTimeScaling __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__msg__SetTimeScaling __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SetTimeScaling_
{
  using Type = SetTimeScaling_<ContainerAllocator>;

  explicit SetTimeScaling_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->steps = 0l;
      this->scaling = 0.0;
    }
  }

  explicit SetTimeScaling_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->steps = 0l;
      this->scaling = 0.0;
    }
  }

  // field types and members
  using _steps_type =
    int32_t;
  _steps_type steps;
  using _scaling_type =
    double;
  _scaling_type scaling;

  // setters for named parameter idiom
  Type & set__steps(
    const int32_t & _arg)
  {
    this->steps = _arg;
    return *this;
  }
  Type & set__scaling(
    const double & _arg)
  {
    this->scaling = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__msg__SetTimeScaling
    std::shared_ptr<panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__msg__SetTimeScaling
    std::shared_ptr<panda_motion_generator_msgs::msg::SetTimeScaling_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetTimeScaling_ & other) const
  {
    if (this->steps != other.steps) {
      return false;
    }
    if (this->scaling != other.scaling) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetTimeScaling_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetTimeScaling_

// alias to use template instance with default allocator
using SetTimeScaling =
  panda_motion_generator_msgs::msg::SetTimeScaling_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace panda_motion_generator_msgs

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SET_TIME_SCALING__STRUCT_HPP_
