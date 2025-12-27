// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:msg/PluginStats.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__msg__PluginStats __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__msg__PluginStats __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PluginStats_
{
  using Type = PluginStats_<ContainerAllocator>;

  explicit PluginStats_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->plugin_type = "";
      this->load_time = 0.0f;
      this->reset_time = 0.0f;
      this->ema_steptime_control = 0.0f;
      this->ema_steptime_passive = 0.0f;
      this->ema_steptime_render = 0.0f;
      this->ema_steptime_last_stage = 0.0f;
    }
  }

  explicit PluginStats_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : plugin_type(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->plugin_type = "";
      this->load_time = 0.0f;
      this->reset_time = 0.0f;
      this->ema_steptime_control = 0.0f;
      this->ema_steptime_passive = 0.0f;
      this->ema_steptime_render = 0.0f;
      this->ema_steptime_last_stage = 0.0f;
    }
  }

  // field types and members
  using _plugin_type_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _plugin_type_type plugin_type;
  using _load_time_type =
    float;
  _load_time_type load_time;
  using _reset_time_type =
    float;
  _reset_time_type reset_time;
  using _ema_steptime_control_type =
    float;
  _ema_steptime_control_type ema_steptime_control;
  using _ema_steptime_passive_type =
    float;
  _ema_steptime_passive_type ema_steptime_passive;
  using _ema_steptime_render_type =
    float;
  _ema_steptime_render_type ema_steptime_render;
  using _ema_steptime_last_stage_type =
    float;
  _ema_steptime_last_stage_type ema_steptime_last_stage;

  // setters for named parameter idiom
  Type & set__plugin_type(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->plugin_type = _arg;
    return *this;
  }
  Type & set__load_time(
    const float & _arg)
  {
    this->load_time = _arg;
    return *this;
  }
  Type & set__reset_time(
    const float & _arg)
  {
    this->reset_time = _arg;
    return *this;
  }
  Type & set__ema_steptime_control(
    const float & _arg)
  {
    this->ema_steptime_control = _arg;
    return *this;
  }
  Type & set__ema_steptime_passive(
    const float & _arg)
  {
    this->ema_steptime_passive = _arg;
    return *this;
  }
  Type & set__ema_steptime_render(
    const float & _arg)
  {
    this->ema_steptime_render = _arg;
    return *this;
  }
  Type & set__ema_steptime_last_stage(
    const float & _arg)
  {
    this->ema_steptime_last_stage = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__msg__PluginStats
    std::shared_ptr<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__msg__PluginStats
    std::shared_ptr<mujoco_ros_msgs::msg::PluginStats_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PluginStats_ & other) const
  {
    if (this->plugin_type != other.plugin_type) {
      return false;
    }
    if (this->load_time != other.load_time) {
      return false;
    }
    if (this->reset_time != other.reset_time) {
      return false;
    }
    if (this->ema_steptime_control != other.ema_steptime_control) {
      return false;
    }
    if (this->ema_steptime_passive != other.ema_steptime_passive) {
      return false;
    }
    if (this->ema_steptime_render != other.ema_steptime_render) {
      return false;
    }
    if (this->ema_steptime_last_stage != other.ema_steptime_last_stage) {
      return false;
    }
    return true;
  }
  bool operator!=(const PluginStats_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PluginStats_

// alias to use template instance with default allocator
using PluginStats =
  mujoco_ros_msgs::msg::PluginStats_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__PLUGIN_STATS__STRUCT_HPP_
