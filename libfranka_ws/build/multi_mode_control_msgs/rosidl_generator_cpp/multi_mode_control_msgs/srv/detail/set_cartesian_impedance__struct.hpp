// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from multi_mode_control_msgs:srv/SetCartesianImpedance.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_CARTESIAN_IMPEDANCE__STRUCT_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_CARTESIAN_IMPEDANCE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__SetCartesianImpedance_Request __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__SetCartesianImpedance_Request __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetCartesianImpedance_Request_
{
  using Type = SetCartesianImpedance_Request_<ContainerAllocator>;

  explicit SetCartesianImpedance_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 36>::iterator, double>(this->stiffness.begin(), this->stiffness.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->damping_ratio.begin(), this->damping_ratio.end(), 0.0);
      this->nullspace_stiffness = 0.0;
    }
  }

  explicit SetCartesianImpedance_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stiffness(_alloc),
    damping_ratio(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 36>::iterator, double>(this->stiffness.begin(), this->stiffness.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->damping_ratio.begin(), this->damping_ratio.end(), 0.0);
      this->nullspace_stiffness = 0.0;
    }
  }

  // field types and members
  using _stiffness_type =
    std::array<double, 36>;
  _stiffness_type stiffness;
  using _damping_ratio_type =
    std::array<double, 6>;
  _damping_ratio_type damping_ratio;
  using _nullspace_stiffness_type =
    double;
  _nullspace_stiffness_type nullspace_stiffness;

  // setters for named parameter idiom
  Type & set__stiffness(
    const std::array<double, 36> & _arg)
  {
    this->stiffness = _arg;
    return *this;
  }
  Type & set__damping_ratio(
    const std::array<double, 6> & _arg)
  {
    this->damping_ratio = _arg;
    return *this;
  }
  Type & set__nullspace_stiffness(
    const double & _arg)
  {
    this->nullspace_stiffness = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetCartesianImpedance_Request
    std::shared_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetCartesianImpedance_Request
    std::shared_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetCartesianImpedance_Request_ & other) const
  {
    if (this->stiffness != other.stiffness) {
      return false;
    }
    if (this->damping_ratio != other.damping_ratio) {
      return false;
    }
    if (this->nullspace_stiffness != other.nullspace_stiffness) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetCartesianImpedance_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetCartesianImpedance_Request_

// alias to use template instance with default allocator
using SetCartesianImpedance_Request =
  multi_mode_control_msgs::srv::SetCartesianImpedance_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs


#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__SetCartesianImpedance_Response __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__SetCartesianImpedance_Response __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetCartesianImpedance_Response_
{
  using Type = SetCartesianImpedance_Response_<ContainerAllocator>;

  explicit SetCartesianImpedance_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit SetCartesianImpedance_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  // field types and members
  using _structure_needs_at_least_one_member_type =
    uint8_t;
  _structure_needs_at_least_one_member_type structure_needs_at_least_one_member;


  // constant declarations

  // pointer types
  using RawPtr =
    multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetCartesianImpedance_Response
    std::shared_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetCartesianImpedance_Response
    std::shared_ptr<multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetCartesianImpedance_Response_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetCartesianImpedance_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetCartesianImpedance_Response_

// alias to use template instance with default allocator
using SetCartesianImpedance_Response =
  multi_mode_control_msgs::srv::SetCartesianImpedance_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs

namespace multi_mode_control_msgs
{

namespace srv
{

struct SetCartesianImpedance
{
  using Request = multi_mode_control_msgs::srv::SetCartesianImpedance_Request;
  using Response = multi_mode_control_msgs::srv::SetCartesianImpedance_Response;
};

}  // namespace srv

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_CARTESIAN_IMPEDANCE__STRUCT_HPP_
