// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:msg/SolverParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__msg__SolverParameters __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__msg__SolverParameters __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SolverParameters_
{
  using Type = SolverParameters_<ContainerAllocator>;

  explicit SolverParameters_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->dmin = 0.0;
      this->dmax = 0.0;
      this->width = 0.0;
      this->midpoint = 0.0;
      this->power = 0.0;
      this->timeconst = 0.0;
      this->dampratio = 0.0;
    }
  }

  explicit SolverParameters_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->dmin = 0.0;
      this->dmax = 0.0;
      this->width = 0.0;
      this->midpoint = 0.0;
      this->power = 0.0;
      this->timeconst = 0.0;
      this->dampratio = 0.0;
    }
  }

  // field types and members
  using _dmin_type =
    double;
  _dmin_type dmin;
  using _dmax_type =
    double;
  _dmax_type dmax;
  using _width_type =
    double;
  _width_type width;
  using _midpoint_type =
    double;
  _midpoint_type midpoint;
  using _power_type =
    double;
  _power_type power;
  using _timeconst_type =
    double;
  _timeconst_type timeconst;
  using _dampratio_type =
    double;
  _dampratio_type dampratio;

  // setters for named parameter idiom
  Type & set__dmin(
    const double & _arg)
  {
    this->dmin = _arg;
    return *this;
  }
  Type & set__dmax(
    const double & _arg)
  {
    this->dmax = _arg;
    return *this;
  }
  Type & set__width(
    const double & _arg)
  {
    this->width = _arg;
    return *this;
  }
  Type & set__midpoint(
    const double & _arg)
  {
    this->midpoint = _arg;
    return *this;
  }
  Type & set__power(
    const double & _arg)
  {
    this->power = _arg;
    return *this;
  }
  Type & set__timeconst(
    const double & _arg)
  {
    this->timeconst = _arg;
    return *this;
  }
  Type & set__dampratio(
    const double & _arg)
  {
    this->dampratio = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__msg__SolverParameters
    std::shared_ptr<mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__msg__SolverParameters
    std::shared_ptr<mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SolverParameters_ & other) const
  {
    if (this->dmin != other.dmin) {
      return false;
    }
    if (this->dmax != other.dmax) {
      return false;
    }
    if (this->width != other.width) {
      return false;
    }
    if (this->midpoint != other.midpoint) {
      return false;
    }
    if (this->power != other.power) {
      return false;
    }
    if (this->timeconst != other.timeconst) {
      return false;
    }
    if (this->dampratio != other.dampratio) {
      return false;
    }
    return true;
  }
  bool operator!=(const SolverParameters_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SolverParameters_

// alias to use template instance with default allocator
using SolverParameters =
  mujoco_ros_msgs::msg::SolverParameters_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SOLVER_PARAMETERS__STRUCT_HPP_
