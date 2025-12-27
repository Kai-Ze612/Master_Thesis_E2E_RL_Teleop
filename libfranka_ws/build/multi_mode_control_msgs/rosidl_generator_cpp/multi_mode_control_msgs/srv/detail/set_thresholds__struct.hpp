// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from multi_mode_control_msgs:srv/SetThresholds.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_THRESHOLDS__STRUCT_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_THRESHOLDS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'level1'
// Member 'level2'
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__SetThresholds_Request __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__SetThresholds_Request __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetThresholds_Request_
{
  using Type = SetThresholds_Request_<ContainerAllocator>;

  explicit SetThresholds_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : level1(_init),
    level2(_init)
  {
    (void)_init;
  }

  explicit SetThresholds_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : level1(_alloc, _init),
    level2(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _level1_type =
    multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator>;
  _level1_type level1;
  using _level2_type =
    multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator>;
  _level2_type level2;

  // setters for named parameter idiom
  Type & set__level1(
    const multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator> & _arg)
  {
    this->level1 = _arg;
    return *this;
  }
  Type & set__level2(
    const multi_mode_control_msgs::msg::ContactThresholds_<ContainerAllocator> & _arg)
  {
    this->level2 = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetThresholds_Request
    std::shared_ptr<multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetThresholds_Request
    std::shared_ptr<multi_mode_control_msgs::srv::SetThresholds_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetThresholds_Request_ & other) const
  {
    if (this->level1 != other.level1) {
      return false;
    }
    if (this->level2 != other.level2) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetThresholds_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetThresholds_Request_

// alias to use template instance with default allocator
using SetThresholds_Request =
  multi_mode_control_msgs::srv::SetThresholds_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs


#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__SetThresholds_Response __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__SetThresholds_Response __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetThresholds_Response_
{
  using Type = SetThresholds_Response_<ContainerAllocator>;

  explicit SetThresholds_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit SetThresholds_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetThresholds_Response
    std::shared_ptr<multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetThresholds_Response
    std::shared_ptr<multi_mode_control_msgs::srv::SetThresholds_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetThresholds_Response_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetThresholds_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetThresholds_Response_

// alias to use template instance with default allocator
using SetThresholds_Response =
  multi_mode_control_msgs::srv::SetThresholds_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs

namespace multi_mode_control_msgs
{

namespace srv
{

struct SetThresholds
{
  using Request = multi_mode_control_msgs::srv::SetThresholds_Request;
  using Response = multi_mode_control_msgs::srv::SetThresholds_Response;
};

}  // namespace srv

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_THRESHOLDS__STRUCT_HPP_
