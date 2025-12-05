// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:srv/GetGeomProperties.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GEOM_PROPERTIES__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GEOM_PROPERTIES__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__GetGeomProperties_Request __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__GetGeomProperties_Request __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetGeomProperties_Request_
{
  using Type = GetGeomProperties_Request_<ContainerAllocator>;

  explicit GetGeomProperties_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->geom_name = "";
      this->admin_hash = "";
    }
  }

  explicit GetGeomProperties_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : geom_name(_alloc),
    admin_hash(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->geom_name = "";
      this->admin_hash = "";
    }
  }

  // field types and members
  using _geom_name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _geom_name_type geom_name;
  using _admin_hash_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _admin_hash_type admin_hash;

  // setters for named parameter idiom
  Type & set__geom_name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->geom_name = _arg;
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
    mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__GetGeomProperties_Request
    std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__GetGeomProperties_Request
    std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetGeomProperties_Request_ & other) const
  {
    if (this->geom_name != other.geom_name) {
      return false;
    }
    if (this->admin_hash != other.admin_hash) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetGeomProperties_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetGeomProperties_Request_

// alias to use template instance with default allocator
using GetGeomProperties_Request =
  mujoco_ros_msgs::srv::GetGeomProperties_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs


// Include directives for member types
// Member 'properties'
#include "mujoco_ros_msgs/msg/detail/geom_properties__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__srv__GetGeomProperties_Response __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__srv__GetGeomProperties_Response __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetGeomProperties_Response_
{
  using Type = GetGeomProperties_Response_<ContainerAllocator>;

  explicit GetGeomProperties_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : properties(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->status_message = "";
    }
  }

  explicit GetGeomProperties_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : properties(_alloc, _init),
    status_message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->status_message = "";
    }
  }

  // field types and members
  using _properties_type =
    mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator>;
  _properties_type properties;
  using _success_type =
    bool;
  _success_type success;
  using _status_message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _status_message_type status_message;

  // setters for named parameter idiom
  Type & set__properties(
    const mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator> & _arg)
  {
    this->properties = _arg;
    return *this;
  }
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
    mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__srv__GetGeomProperties_Response
    std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__srv__GetGeomProperties_Response
    std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetGeomProperties_Response_ & other) const
  {
    if (this->properties != other.properties) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    if (this->status_message != other.status_message) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetGeomProperties_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetGeomProperties_Response_

// alias to use template instance with default allocator
using GetGeomProperties_Response =
  mujoco_ros_msgs::srv::GetGeomProperties_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace mujoco_ros_msgs

namespace mujoco_ros_msgs
{

namespace srv
{

struct GetGeomProperties
{
  using Request = mujoco_ros_msgs::srv::GetGeomProperties_Request;
  using Response = mujoco_ros_msgs::srv::GetGeomProperties_Response;
};

}  // namespace srv

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__SRV__DETAIL__GET_GEOM_PROPERTIES__STRUCT_HPP_
