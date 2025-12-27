// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:srv/SetBodyState.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__SET_BODY_STATE__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__SET_BODY_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'state'
#include "mujoco_ros_msgs/msg/detail/body_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__SetBodyState_Request __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__SetBodyState_Request __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetBodyState_Request_
{
  using Type = SetBodyState_Request_<ContainerAllocator>;

  explicit SetBodyState_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : state(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->set_pose = false;
      this->set_twist = false;
      this->set_mass = false;
      this->reset_qpos = false;
      this->admin_hash = "";
    }
  }

  explicit SetBodyState_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : state(_alloc, _init),
    admin_hash(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->set_pose = false;
      this->set_twist = false;
      this->set_mass = false;
      this->reset_qpos = false;
      this->admin_hash = "";
    }
  }

  // field types and members
  using _state_type =
    mujoco_ros_msgs::msg::BodyState_<ContainerAllocator>;
  _state_type state;
  using _set_pose_type =
    bool;
  _set_pose_type set_pose;
  using _set_twist_type =
    bool;
  _set_twist_type set_twist;
  using _set_mass_type =
    bool;
  _set_mass_type set_mass;
  using _reset_qpos_type =
    bool;
  _reset_qpos_type reset_qpos;
  using _admin_hash_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _admin_hash_type admin_hash;

  // setters for named parameter idiom
  Type & set__state(
    const mujoco_ros_msgs::msg::BodyState_<ContainerAllocator> & _arg)
  {
    this->state = _arg;
    return *this;
  }
  Type & set__set_pose(
    const bool & _arg)
  {
    this->set_pose = _arg;
    return *this;
  }
  Type & set__set_twist(
    const bool & _arg)
  {
    this->set_twist = _arg;
    return *this;
  }
  Type & set__set_mass(
    const bool & _arg)
  {
    this->set_mass = _arg;
    return *this;
  }
  Type & set__reset_qpos(
    const bool & _arg)
  {
    this->reset_qpos = _arg;
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
    mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__SetBodyState_Request
    std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__SetBodyState_Request
    std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetBodyState_Request_ & other) const
  {
    if (this->state != other.state) {
      return false;
    }
    if (this->set_pose != other.set_pose) {
      return false;
    }
    if (this->set_twist != other.set_twist) {
      return false;
    }
    if (this->set_mass != other.set_mass) {
      return false;
    }
    if (this->reset_qpos != other.reset_qpos) {
      return false;
    }
    if (this->admin_hash != other.admin_hash) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetBodyState_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetBodyState_Request_

// alias to use template instance with default allocator
using SetBodyState_Request =
  mujoco_ros_msgs::srv::SetBodyState_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__SetBodyState_Response __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__SetBodyState_Response __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetBodyState_Response_
{
  using Type = SetBodyState_Response_<ContainerAllocator>;

  explicit SetBodyState_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->status_message = "";
    }
  }

  explicit SetBodyState_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : status_message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->status_message = "";
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;
  using _status_message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _status_message_type status_message;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__status_message(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->status_message = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__SetBodyState_Response
    std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__SetBodyState_Response
    std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetBodyState_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->status_message != other.status_message) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetBodyState_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetBodyState_Response_

// alias to use template instance with default allocator
using SetBodyState_Response =
  mujoco_ros_msgs::srv::SetBodyState_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs

namespace mujoco_ros_msgs
{

namespace srv
{

struct SetBodyState
{
  using Request = mujoco_ros_msgs::srv::SetBodyState_Request;
  using Response = mujoco_ros_msgs::srv::SetBodyState_Response;
};

}  // namespace srv

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__SET_BODY_STATE__STRUCT_HPP_
