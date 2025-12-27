// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:msg/SimInfo.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'loading_state'
#include "mujoco_ros_msgs/msg/detail/state_uint__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__msg__SimInfo __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__msg__SimInfo __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SimInfo_
{
  using Type = SimInfo_<ContainerAllocator>;

  explicit SimInfo_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : loading_state(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->model_path = "";
      this->model_valid = false;
      this->load_count = 0;
      this->paused = false;
      this->pending_sim_steps = 0;
      this->rt_measured = 0.0f;
      this->rt_setting = 0.0f;
    }
  }

  explicit SimInfo_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : model_path(_alloc),
    loading_state(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->model_path = "";
      this->model_valid = false;
      this->load_count = 0;
      this->paused = false;
      this->pending_sim_steps = 0;
      this->rt_measured = 0.0f;
      this->rt_setting = 0.0f;
    }
  }

  // field types and members
  using _model_path_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _model_path_type model_path;
  using _model_valid_type =
    bool;
  _model_valid_type model_valid;
  using _load_count_type =
    uint16_t;
  _load_count_type load_count;
  using _loading_state_type =
    mujoco_ros_msgs::msg::StateUint_<ContainerAllocator>;
  _loading_state_type loading_state;
  using _paused_type =
    bool;
  _paused_type paused;
  using _pending_sim_steps_type =
    uint16_t;
  _pending_sim_steps_type pending_sim_steps;
  using _rt_measured_type =
    float;
  _rt_measured_type rt_measured;
  using _rt_setting_type =
    float;
  _rt_setting_type rt_setting;

  // setters for named parameter idiom
  Type & set__model_path(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->model_path = _arg;
    return *this;
  }
  Type & set__model_valid(
    const bool & _arg)
  {
    this->model_valid = _arg;
    return *this;
  }
  Type & set__load_count(
    const uint16_t & _arg)
  {
    this->load_count = _arg;
    return *this;
  }
  Type & set__loading_state(
    const mujoco_ros_msgs::msg::StateUint_<ContainerAllocator> & _arg)
  {
    this->loading_state = _arg;
    return *this;
  }
  Type & set__paused(
    const bool & _arg)
  {
    this->paused = _arg;
    return *this;
  }
  Type & set__pending_sim_steps(
    const uint16_t & _arg)
  {
    this->pending_sim_steps = _arg;
    return *this;
  }
  Type & set__rt_measured(
    const float & _arg)
  {
    this->rt_measured = _arg;
    return *this;
  }
  Type & set__rt_setting(
    const float & _arg)
  {
    this->rt_setting = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__msg__SimInfo
    std::shared_ptr<mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__msg__SimInfo
    std::shared_ptr<mujoco_ros_msgs::msg::SimInfo_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SimInfo_ & other) const
  {
    if (this->model_path != other.model_path) {
      return false;
    }
    if (this->model_valid != other.model_valid) {
      return false;
    }
    if (this->load_count != other.load_count) {
      return false;
    }
    if (this->loading_state != other.loading_state) {
      return false;
    }
    if (this->paused != other.paused) {
      return false;
    }
    if (this->pending_sim_steps != other.pending_sim_steps) {
      return false;
    }
    if (this->rt_measured != other.rt_measured) {
      return false;
    }
    if (this->rt_setting != other.rt_setting) {
      return false;
    }
    return true;
  }
  bool operator!=(const SimInfo_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SimInfo_

// alias to use template instance with default allocator
using SimInfo =
  mujoco_ros_msgs::msg::SimInfo_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SIM_INFO__STRUCT_HPP_
