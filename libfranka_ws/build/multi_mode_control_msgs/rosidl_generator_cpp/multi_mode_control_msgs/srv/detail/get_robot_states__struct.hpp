// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from multi_mode_control_msgs:srv/GetRobotStates.idl
// generated code does not contain a copyright notice

#ifndef MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__GET_ROBOT_STATES__STRUCT_HPP_
#define MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__GET_ROBOT_STATES__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__GetRobotStates_Request __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__GetRobotStates_Request __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetRobotStates_Request_
{
  using Type = GetRobotStates_Request_<ContainerAllocator>;

  explicit GetRobotStates_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit GetRobotStates_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__GetRobotStates_Request
    std::shared_ptr<multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__GetRobotStates_Request
    std::shared_ptr<multi_mode_control_msgs::srv::GetRobotStates_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetRobotStates_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetRobotStates_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetRobotStates_Request_

// alias to use template instance with default allocator
using GetRobotStates_Request =
  multi_mode_control_msgs::srv::GetRobotStates_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs


// Include directives for member types
// Member 'states'
#include "franka_msgs/msg/detail/franka_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__multi_mode_control_msgs__srv__GetRobotStates_Response __attribute__((deprecated))
#else
# define DEPRECATED__multi_mode_control_msgs__srv__GetRobotStates_Response __declspec(deprecated)
#endif

namespace multi_mode_control_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetRobotStates_Response_
{
  using Type = GetRobotStates_Response_<ContainerAllocator>;

  explicit GetRobotStates_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit GetRobotStates_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _states_type =
    std::vector<franka_msgs::msg::FrankaState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<franka_msgs::msg::FrankaState_<ContainerAllocator>>>;
  _states_type states;

  // setters for named parameter idiom
  Type & set__states(
    const std::vector<franka_msgs::msg::FrankaState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<franka_msgs::msg::FrankaState_<ContainerAllocator>>> & _arg)
  {
    this->states = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__multi_mode_control_msgs__srv__GetRobotStates_Response
    std::shared_ptr<multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__multi_mode_control_msgs__srv__GetRobotStates_Response
    std::shared_ptr<multi_mode_control_msgs::srv::GetRobotStates_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetRobotStates_Response_ & other) const
  {
    if (this->states != other.states) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetRobotStates_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetRobotStates_Response_

// alias to use template instance with default allocator
using GetRobotStates_Response =
  multi_mode_control_msgs::srv::GetRobotStates_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace multi_mode_control_msgs

namespace multi_mode_control_msgs
{

namespace srv
{

struct GetRobotStates
{
  using Request = multi_mode_control_msgs::srv::GetRobotStates_Request;
  using Response = multi_mode_control_msgs::srv::GetRobotStates_Response;
};

}  // namespace srv

}  // namespace multi_mode_control_msgs

#endif  // MULTI_MODE_CONTROL_MSGS__SRV__DETAIL__GET_ROBOT_STATES__STRUCT_HPP_
