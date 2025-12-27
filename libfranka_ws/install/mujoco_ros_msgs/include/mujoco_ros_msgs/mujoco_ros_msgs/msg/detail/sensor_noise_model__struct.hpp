// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:msg/SensorNoiseModel.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__msg__SensorNoiseModel __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__msg__SensorNoiseModel __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SensorNoiseModel_
{
  using Type = SensorNoiseModel_<ContainerAllocator>;

  explicit SensorNoiseModel_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sensor_name = "";
      this->set_flag = 0;
    }
  }

  explicit SensorNoiseModel_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : sensor_name(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sensor_name = "";
      this->set_flag = 0;
    }
  }

  // field types and members
  using _sensor_name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _sensor_name_type sensor_name;
  using _mean_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _mean_type mean;
  using _std_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _std_type std;
  using _set_flag_type =
    uint8_t;
  _set_flag_type set_flag;

  // setters for named parameter idiom
  Type & set__sensor_name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->sensor_name = _arg;
    return *this;
  }
  Type & set__mean(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->mean = _arg;
    return *this;
  }
  Type & set__std(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->std = _arg;
    return *this;
  }
  Type & set__set_flag(
    const uint8_t & _arg)
  {
    this->set_flag = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__msg__SensorNoiseModel
    std::shared_ptr<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__msg__SensorNoiseModel
    std::shared_ptr<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SensorNoiseModel_ & other) const
  {
    if (this->sensor_name != other.sensor_name) {
      return false;
    }
    if (this->mean != other.mean) {
      return false;
    }
    if (this->std != other.std) {
      return false;
    }
    if (this->set_flag != other.set_flag) {
      return false;
    }
    return true;
  }
  bool operator!=(const SensorNoiseModel_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SensorNoiseModel_

// alias to use template instance with default allocator
using SensorNoiseModel =
  mujoco_ros_msgs::msg::SensorNoiseModel_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__SENSOR_NOISE_MODEL__STRUCT_HPP_
