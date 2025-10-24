// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:srv/GetPluginStats.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_PLUGIN_STATS__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_PLUGIN_STATS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__GetPluginStats_Request __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__GetPluginStats_Request __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetPluginStats_Request_
{
  using Type = GetPluginStats_Request_<ContainerAllocator>;

  explicit GetPluginStats_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit GetPluginStats_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__GetPluginStats_Request
    std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__GetPluginStats_Request
    std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetPluginStats_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetPluginStats_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetPluginStats_Request_

// alias to use template instance with default allocator
using GetPluginStats_Request =
  mujoco_ros_msgs::srv::GetPluginStats_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs


// Include directives for member types
// Member 'stats'
#include "mujoco_ros_msgs/msg/detail/plugin_stats__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__GetPluginStats_Response __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__GetPluginStats_Response __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetPluginStats_Response_
{
  using Type = GetPluginStats_Response_<ContainerAllocator>;

  explicit GetPluginStats_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit GetPluginStats_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _stats_type =
    std::vector<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>>>;
  _stats_type stats;

  // setters for named parameter idiom
  Type & set__stats(
    const std::vector<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>>> & _arg)
  {
    this->stats = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__GetPluginStats_Response
    std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__GetPluginStats_Response
    std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetPluginStats_Response_ & other) const
  {
    if (this->stats != other.stats) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetPluginStats_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetPluginStats_Response_

// alias to use template instance with default allocator
using GetPluginStats_Response =
  mujoco_ros_msgs::srv::GetPluginStats_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs

namespace mujoco_ros_msgs
{

namespace srv
{

struct GetPluginStats
{
  using Request = mujoco_ros_msgs::srv::GetPluginStats_Request;
  using Response = mujoco_ros_msgs::srv::GetPluginStats_Response;
};

}  // namespace srv

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_PLUGIN_STATS__STRUCT_HPP_
