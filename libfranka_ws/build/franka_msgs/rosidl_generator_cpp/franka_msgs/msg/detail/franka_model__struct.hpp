// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from franka_msgs:msg/FrankaModel.idl
// generated code does not contain a copyright notice

#ifndef FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__STRUCT_HPP_
#define FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__franka_msgs__msg__FrankaModel __attribute__((deprecated))
#else
# define DEPRECATED__franka_msgs__msg__FrankaModel __declspec(deprecated)
#endif

namespace franka_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct FrankaModel_
{
  using Type = FrankaModel_<ContainerAllocator>;

  explicit FrankaModel_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 7>::iterator, double>(this->coriolis.begin(), this->coriolis.end(), 0.0);
      std::fill<typename std::array<double, 42>::iterator, double>(this->ee_zero_jacobian.begin(), this->ee_zero_jacobian.end(), 0.0);
      std::fill<typename std::array<double, 42>::iterator, double>(this->ee_body_jacobian.begin(), this->ee_body_jacobian.end(), 0.0);
      std::fill<typename std::array<double, 49>::iterator, double>(this->mass.begin(), this->mass.end(), 0.0);
    }
  }

  explicit FrankaModel_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    coriolis(_alloc),
    ee_zero_jacobian(_alloc),
    ee_body_jacobian(_alloc),
    mass(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 7>::iterator, double>(this->coriolis.begin(), this->coriolis.end(), 0.0);
      std::fill<typename std::array<double, 42>::iterator, double>(this->ee_zero_jacobian.begin(), this->ee_zero_jacobian.end(), 0.0);
      std::fill<typename std::array<double, 42>::iterator, double>(this->ee_body_jacobian.begin(), this->ee_body_jacobian.end(), 0.0);
      std::fill<typename std::array<double, 49>::iterator, double>(this->mass.begin(), this->mass.end(), 0.0);
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _coriolis_type =
    std::array<double, 7>;
  _coriolis_type coriolis;
  using _ee_zero_jacobian_type =
    std::array<double, 42>;
  _ee_zero_jacobian_type ee_zero_jacobian;
  using _ee_body_jacobian_type =
    std::array<double, 42>;
  _ee_body_jacobian_type ee_body_jacobian;
  using _mass_type =
    std::array<double, 49>;
  _mass_type mass;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__coriolis(
    const std::array<double, 7> & _arg)
  {
    this->coriolis = _arg;
    return *this;
  }
  Type & set__ee_zero_jacobian(
    const std::array<double, 42> & _arg)
  {
    this->ee_zero_jacobian = _arg;
    return *this;
  }
  Type & set__ee_body_jacobian(
    const std::array<double, 42> & _arg)
  {
    this->ee_body_jacobian = _arg;
    return *this;
  }
  Type & set__mass(
    const std::array<double, 49> & _arg)
  {
    this->mass = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    franka_msgs::msg::FrankaModel_<ContainerAllocator> *;
  using ConstRawPtr =
    const franka_msgs::msg::FrankaModel_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<franka_msgs::msg::FrankaModel_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<franka_msgs::msg::FrankaModel_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      franka_msgs::msg::FrankaModel_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<franka_msgs::msg::FrankaModel_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      franka_msgs::msg::FrankaModel_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<franka_msgs::msg::FrankaModel_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<franka_msgs::msg::FrankaModel_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<franka_msgs::msg::FrankaModel_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__franka_msgs__msg__FrankaModel
    std::shared_ptr<franka_msgs::msg::FrankaModel_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__franka_msgs__msg__FrankaModel
    std::shared_ptr<franka_msgs::msg::FrankaModel_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const FrankaModel_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->coriolis != other.coriolis) {
      return false;
    }
    if (this->ee_zero_jacobian != other.ee_zero_jacobian) {
      return false;
    }
    if (this->ee_body_jacobian != other.ee_body_jacobian) {
      return false;
    }
    if (this->mass != other.mass) {
      return false;
    }
    return true;
  }
  bool operator!=(const FrankaModel_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct FrankaModel_

// alias to use template instance with default allocator
using FrankaModel =
  franka_msgs::msg::FrankaModel_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace franka_msgs

#endif  // FRANKA_MSGS__MSG__DETAIL__FRANKA_MODEL__STRUCT_HPP_
