// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from panda_motion_generator_msgs:msg/SimpleActionResult.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__STRUCT_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__msg__SimpleActionResult __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__msg__SimpleActionResult __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SimpleActionResult_
{
  using Type = SimpleActionResult_<ContainerAllocator>;

  explicit SimpleActionResult_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->state = 0l;
      this->error = "";
    }
  }

  explicit SimpleActionResult_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : error(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->state = 0l;
      this->error = "";
    }
  }

  // field types and members
  using _state_type =
    int32_t;
  _state_type state;
  using _error_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _error_type error;

  // setters for named parameter idiom
  Type & set__state(
    const int32_t & _arg)
  {
    this->state = _arg;
    return *this;
  }
  Type & set__error(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->error = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int32_t SUCCESS =
    0;
  static constexpr int32_t FAILURE =
    1;
  static constexpr int32_t REJECTED =
    2;
  static constexpr int32_t ABORTED =
    3;
  static constexpr int32_t PREEMPTED =
    4;

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__msg__SimpleActionResult
    std::shared_ptr<panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__msg__SimpleActionResult
    std::shared_ptr<panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SimpleActionResult_ & other) const
  {
    if (this->state != other.state) {
      return false;
    }
    if (this->error != other.error) {
      return false;
    }
    return true;
  }
  bool operator!=(const SimpleActionResult_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SimpleActionResult_

// alias to use template instance with default allocator
using SimpleActionResult =
  panda_motion_generator_msgs::msg::SimpleActionResult_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int32_t SimpleActionResult_<ContainerAllocator>::SUCCESS;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int32_t SimpleActionResult_<ContainerAllocator>::FAILURE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int32_t SimpleActionResult_<ContainerAllocator>::REJECTED;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int32_t SimpleActionResult_<ContainerAllocator>::ABORTED;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int32_t SimpleActionResult_<ContainerAllocator>::PREEMPTED;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace panda_motion_generator_msgs

#endif  // PANDA_MOTION_GENERATOR_MSGS__MSG__DETAIL__SIMPLE_ACTION_RESULT__STRUCT_HPP_
