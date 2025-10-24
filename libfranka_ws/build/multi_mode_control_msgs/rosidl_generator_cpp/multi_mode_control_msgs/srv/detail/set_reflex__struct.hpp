// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from multi_mode_control_msgs:srv/SetReflex.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_REFLEX__STRUCT_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_REFLEX__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__SetReflex_Request __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__SetReflex_Request __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetReflex_Request_
{
  using Type = SetReflex_Request_<ContainerAllocator>;

  explicit SetReflex_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->level1 = "";
      this->level2 = "";
    }
  }

  explicit SetReflex_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : level1(_alloc),
    level2(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->level1 = "";
      this->level2 = "";
    }
  }

  // field types and members
  using _level1_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _level1_type level1;
  using _level2_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _level2_type level2;

  // setters for named parameter idiom
  Type & set__level1(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->level1 = _arg;
    return *this;
  }
  Type & set__level2(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->level2 = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetReflex_Request
    std::shared_ptr<multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetReflex_Request
    std::shared_ptr<multi_mode_control_msgs::srv::SetReflex_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetReflex_Request_ & other) const
  {
    if (this->level1 != other.level1) {
      return false;
    }
    if (this->level2 != other.level2) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetReflex_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetReflex_Request_

// alias to use template instance with default allocator
using SetReflex_Request =
  multi_mode_control_msgs::srv::SetReflex_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs


#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__SetReflex_Response __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__SetReflex_Response __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetReflex_Response_
{
  using Type = SetReflex_Response_<ContainerAllocator>;

  explicit SetReflex_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit SetReflex_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetReflex_Response
    std::shared_ptr<multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__SetReflex_Response
    std::shared_ptr<multi_mode_control_msgs::srv::SetReflex_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetReflex_Response_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetReflex_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetReflex_Response_

// alias to use template instance with default allocator
using SetReflex_Response =
  multi_mode_control_msgs::srv::SetReflex_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs

namespace multi_mode_control_msgs
{

namespace srv
{

struct SetReflex
{
  using Request = multi_mode_control_msgs::srv::SetReflex_Request;
  using Response = multi_mode_control_msgs::srv::SetReflex_Response;
};

}  // namespace srv

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__SET_REFLEX__STRUCT_HPP_
