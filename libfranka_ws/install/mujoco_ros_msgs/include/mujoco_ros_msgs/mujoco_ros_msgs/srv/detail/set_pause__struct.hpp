// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:srv/SetPause.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__SET_PAUSE__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__SET_PAUSE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__SetPause_Request __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__SetPause_Request __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetPause_Request_
{
  using Type = SetPause_Request_<ContainerAllocator>;

  explicit SetPause_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->paused = false;
      this->admin_hash = "";
    }
  }

  explicit SetPause_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : admin_hash(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->paused = false;
      this->admin_hash = "";
    }
  }

  // field types and members
  using _paused_type =
    bool;
  _paused_type paused;
  using _admin_hash_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _admin_hash_type admin_hash;

  // setters for named parameter idiom
  Type & set__paused(
    const bool & _arg)
  {
    this->paused = _arg;
    return *this;
  }
  Type & set__admin_hash(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->admin_hash = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__SetPause_Request
    std::shared_ptr<mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__SetPause_Request
    std::shared_ptr<mujoco_ros_msgs::srv::SetPause_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetPause_Request_ & other) const
  {
    if (this->paused != other.paused) {
      return false;
    }
    if (this->admin_hash != other.admin_hash) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetPause_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetPause_Request_

// alias to use template instance with default allocator
using SetPause_Request =
  mujoco_ros_msgs::srv::SetPause_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__SetPause_Response __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__SetPause_Response __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetPause_Response_
{
  using Type = SetPause_Response_<ContainerAllocator>;

  explicit SetPause_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit SetPause_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__SetPause_Response
    std::shared_ptr<mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__SetPause_Response
    std::shared_ptr<mujoco_ros_msgs::srv::SetPause_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetPause_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetPause_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetPause_Response_

// alias to use template instance with default allocator
using SetPause_Response =
  mujoco_ros_msgs::srv::SetPause_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs

namespace mujoco_ros_msgs
{

namespace srv
{

struct SetPause
{
  using Request = mujoco_ros_msgs::srv::SetPause_Request;
  using Response = mujoco_ros_msgs::srv::SetPause_Response;
};

}  // namespace srv

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__SET_PAUSE__STRUCT_HPP_
