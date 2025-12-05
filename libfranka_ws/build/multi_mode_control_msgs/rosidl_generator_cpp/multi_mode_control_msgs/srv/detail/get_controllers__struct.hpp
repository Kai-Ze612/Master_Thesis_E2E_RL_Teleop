// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from multi_mode_control_msgs:srv/GetControllers.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__GET_CONTROLLERS__STRUCT_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__GET_CONTROLLERS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__GetControllers_Request __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__GetControllers_Request __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetControllers_Request_
{
  using Type = GetControllers_Request_<ContainerAllocator>;

  explicit GetControllers_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit GetControllers_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__GetControllers_Request
    std::shared_ptr<multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__GetControllers_Request
    std::shared_ptr<multi_mode_control_msgs::srv::GetControllers_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetControllers_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetControllers_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetControllers_Request_

// alias to use template instance with default allocator
using GetControllers_Request =
  multi_mode_control_msgs::srv::GetControllers_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs


// Include directives for member types
// Member 'controllers'
#include "multi_mode_control_msgs/msg/detail/controller__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__GetControllers_Response __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__GetControllers_Response __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetControllers_Response_
{
  using Type = GetControllers_Response_<ContainerAllocator>;

  explicit GetControllers_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit GetControllers_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _controllers_type =
    std::vector<multi_mode_control_msgs::msg::Controller_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<multi_mode_control_msgs::msg::Controller_<ContainerAllocator>>>;
  _controllers_type controllers;

  // setters for named parameter idiom
  Type & set__controllers(
    const std::vector<multi_mode_control_msgs::msg::Controller_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<multi_mode_control_msgs::msg::Controller_<ContainerAllocator>>> & _arg)
  {
    this->controllers = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__GetControllers_Response
    std::shared_ptr<multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__GetControllers_Response
    std::shared_ptr<multi_mode_control_msgs::srv::GetControllers_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetControllers_Response_ & other) const
  {
    if (this->controllers != other.controllers) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetControllers_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetControllers_Response_

// alias to use template instance with default allocator
using GetControllers_Response =
  multi_mode_control_msgs::srv::GetControllers_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs

namespace multi_mode_control_msgs
{

namespace srv
{

struct GetControllers
{
  using Request = multi_mode_control_msgs::srv::GetControllers_Request;
  using Response = multi_mode_control_msgs::srv::GetControllers_Response;
};

}  // namespace srv

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__GET_CONTROLLERS__STRUCT_HPP_
