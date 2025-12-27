// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from multi_mode_control_msgs:msg/ContactThresholds.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__STRUCT_HPP_
#define MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'tau_max'
// Member 'tau_min'
#include "multi_mode_control_msgs/msg/detail/joint_array__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__msg__ContactThresholds __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__msg__ContactThresholds __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ContactThresholds_
{
  using Type = ContactThresholds_<ContainerAllocator>;

  explicit ContactThresholds_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_max.begin(), this->f_max.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_min.begin(), this->f_min.end(), 0.0);
      this->f_abs_max = 0.0;
      this->m_abs_max = 0.0;
      this->abs_mode = false;
      this->directional_mode = false;
      this->torque_mode = false;
    }
  }

  explicit ContactThresholds_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : f_max(_alloc),
    f_min(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_max.begin(), this->f_max.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_min.begin(), this->f_min.end(), 0.0);
      this->f_abs_max = 0.0;
      this->m_abs_max = 0.0;
      this->abs_mode = false;
      this->directional_mode = false;
      this->torque_mode = false;
    }
  }

  // field types and members
  using _f_max_type =
    std::array<double, 6>;
  _f_max_type f_max;
  using _f_min_type =
    std::array<double, 6>;
  _f_min_type f_min;
  using _tau_max_type =
    std::vector<multi_mode_control_msgs::msg::JointArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<multi_mode_control_msgs::msg::JointArray_<ContainerAllocator>>>;
  _tau_max_type tau_max;
  using _tau_min_type =
    std::vector<multi_mode_control_msgs::msg::JointArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<multi_mode_control_msgs::msg::JointArray_<ContainerAllocator>>>;
  _tau_min_type tau_min;
  using _f_abs_max_type =
    double;
  _f_abs_max_type f_abs_max;
  using _m_abs_max_type =
    double;
  _m_abs_max_type m_abs_max;
  using _abs_mode_type =
    bool;
  _abs_mode_type abs_mode;
  using _directional_mode_type =
    bool;
  _directional_mode_type directional_mode;
  using _torque_mode_type =
    bool;
  _torque_mode_type torque_mode;

  // setters for named parameter idiom
  Type & set__f_max(
    const std::array<double, 6> & _arg)
  {
    this->f_max = _arg;
    return *this;
  }
  Type & set__f_min(
    const std::array<double, 6> & _arg)
  {
    this->f_min = _arg;
    return *this;
  }
  Type & set__tau_max(
    const std::vector<multi_mode_control_msgs::msg::JointArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<multi_mode_control_msgs::msg::JointArray_<ContainerAllocator>>> & _arg)
  {
    this->tau_max = _arg;
    return *this;
  }
  Type & set__tau_min(
    const std::vector<multi_mode_control_msgs::msg::JointArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<multi_mode_control_msgs::msg::JointArray_<ContainerAllocator>>> & _arg)
  {
    this->tau_min = _arg;
    return *this;
  }
  Type & set__f_abs_max(
    const double & _arg)
  {
    this->f_abs_max = _arg;
    return *this;
  }
  Type & set__m_abs_max(
    const double & _arg)
  {
    this->m_abs_max = _arg;
    return *this;
  }
  Type & set__abs_mode(
    const bool & _arg)
  {
    this->abs_mode = _arg;
    return *this;
  }
  Type & set__directional_mode(
    const bool & _arg)
  {
    this->directional_mode = _arg;
    return *this;
  }
  Type & set__torque_mode(
    const bool & _arg)
  {
    this->torque_mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__msg__ContactThresholds
    std::shared_ptr<multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__msg__ContactThresholds
    std::shared_ptr<multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ContactThresholds_ & other) const
  {
    if (this->f_max != other.f_max) {
      return false;
    }
    if (this->f_min != other.f_min) {
      return false;
    }
    if (this->tau_max != other.tau_max) {
      return false;
    }
    if (this->tau_min != other.tau_min) {
      return false;
    }
    if (this->f_abs_max != other.f_abs_max) {
      return false;
    }
    if (this->m_abs_max != other.m_abs_max) {
      return false;
    }
    if (this->abs_mode != other.abs_mode) {
      return false;
    }
    if (this->directional_mode != other.directional_mode) {
      return false;
    }
    if (this->torque_mode != other.torque_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const ContactThresholds_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ContactThresholds_

// alias to use template instance with default allocator
using ContactThresholds =
  multi_mode_control_msgs::msg::ContactThresholds_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__MSG__DETAIL__CONTACT_THRESHOLDS__STRUCT_HPP_
