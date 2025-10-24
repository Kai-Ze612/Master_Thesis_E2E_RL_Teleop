// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:srv/RegisterSensorNoiseModels.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__REGISTER_SENSOR_NOISE_MODELS__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__REGISTER_SENSOR_NOISE_MODELS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'noise_models'
#include "mujoco_ros_msgs/msg/detail/sensor_noise_model__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct RegisterSensorNoiseModels_Request_
{
  using Type = RegisterSensorNoiseModels_Request_<ContainerAllocator>;

  explicit RegisterSensorNoiseModels_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->admin_hash = "";
    }
  }

  explicit RegisterSensorNoiseModels_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : admin_hash(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->admin_hash = "";
    }
  }

  // field types and members
  using _noise_models_type =
    std::vector<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>>>;
  _noise_models_type noise_models;
  using _admin_hash_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _admin_hash_type admin_hash;

  // setters for named parameter idiom
  Type & set__noise_models(
    const std::vector<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<mujoco_ros_msgs::msg::SensorNoiseModel_<ContainerAllocator>>> & _arg)
  {
    this->noise_models = _arg;
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
    mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request
    std::shared_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Request
    std::shared_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RegisterSensorNoiseModels_Request_ & other) const
  {
    if (this->noise_models != other.noise_models) {
      return false;
    }
    if (this->admin_hash != other.admin_hash) {
      return false;
    }
    return true;
  }
  bool operator!=(const RegisterSensorNoiseModels_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RegisterSensorNoiseModels_Request_

// alias to use template instance with default allocator
using RegisterSensorNoiseModels_Request =
  mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct RegisterSensorNoiseModels_Response_
{
  using Type = RegisterSensorNoiseModels_Response_<ContainerAllocator>;

  explicit RegisterSensorNoiseModels_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit RegisterSensorNoiseModels_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response
    std::shared_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__RegisterSensorNoiseModels_Response
    std::shared_ptr<mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RegisterSensorNoiseModels_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const RegisterSensorNoiseModels_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RegisterSensorNoiseModels_Response_

// alias to use template instance with default allocator
using RegisterSensorNoiseModels_Response =
  mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs

namespace mujoco_ros_msgs
{

namespace srv
{

struct RegisterSensorNoiseModels
{
  using Request = mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Request;
  using Response = mujoco_ros_msgs::srv::RegisterSensorNoiseModels_Response;
};

}  // namespace srv

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__REGISTER_SENSOR_NOISE_MODELS__STRUCT_HPP_
