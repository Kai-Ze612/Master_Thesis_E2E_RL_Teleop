// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:msg/GeomProperties.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'type'
#include "mujoco_ros_msgs/msg/detail/geom_type__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__msg__GeomProperties __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__msg__GeomProperties __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct GeomProperties_
{
  using Type = GeomProperties_<ContainerAllocator>;

  explicit GeomProperties_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : type(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->name = "";
      this->body_mass = 0.0f;
      this->size_0 = 0.0f;
      this->size_1 = 0.0f;
      this->size_2 = 0.0f;
      this->friction_slide = 0.0f;
      this->friction_spin = 0.0f;
      this->friction_roll = 0.0f;
    }
  }

  explicit GeomProperties_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : name(_alloc),
    type(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->name = "";
      this->body_mass = 0.0f;
      this->size_0 = 0.0f;
      this->size_1 = 0.0f;
      this->size_2 = 0.0f;
      this->friction_slide = 0.0f;
      this->friction_spin = 0.0f;
      this->friction_roll = 0.0f;
    }
  }

  // field types and members
  using _name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _name_type name;
  using _type_type =
    mujoco_ros_msgs::msg::GeomType_<ContainerAllocator>;
  _type_type type;
  using _body_mass_type =
    float;
  _body_mass_type body_mass;
  using _size_0_type =
    float;
  _size_0_type size_0;
  using _size_1_type =
    float;
  _size_1_type size_1;
  using _size_2_type =
    float;
  _size_2_type size_2;
  using _friction_slide_type =
    float;
  _friction_slide_type friction_slide;
  using _friction_spin_type =
    float;
  _friction_spin_type friction_spin;
  using _friction_roll_type =
    float;
  _friction_roll_type friction_roll;

  // setters for named parameter idiom
  Type & set__name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->name = _arg;
    return *this;
  }
  Type & set__type(
    const mujoco_ros_msgs::msg::GeomType_<ContainerAllocator> & _arg)
  {
    this->type = _arg;
    return *this;
  }
  Type & set__body_mass(
    const float & _arg)
  {
    this->body_mass = _arg;
    return *this;
  }
  Type & set__size_0(
    const float & _arg)
  {
    this->size_0 = _arg;
    return *this;
  }
  Type & set__size_1(
    const float & _arg)
  {
    this->size_1 = _arg;
    return *this;
  }
  Type & set__size_2(
    const float & _arg)
  {
    this->size_2 = _arg;
    return *this;
  }
  Type & set__friction_slide(
    const float & _arg)
  {
    this->friction_slide = _arg;
    return *this;
  }
  Type & set__friction_spin(
    const float & _arg)
  {
    this->friction_spin = _arg;
    return *this;
  }
  Type & set__friction_roll(
    const float & _arg)
  {
    this->friction_roll = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__msg__GeomProperties
    std::shared_ptr<mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__msg__GeomProperties
    std::shared_ptr<mujoco_ros_msgs::msg::GeomProperties_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GeomProperties_ & other) const
  {
    if (this->name != other.name) {
      return false;
    }
    if (this->type != other.type) {
      return false;
    }
    if (this->body_mass != other.body_mass) {
      return false;
    }
    if (this->size_0 != other.size_0) {
      return false;
    }
    if (this->size_1 != other.size_1) {
      return false;
    }
    if (this->size_2 != other.size_2) {
      return false;
    }
    if (this->friction_slide != other.friction_slide) {
      return false;
    }
    if (this->friction_spin != other.friction_spin) {
      return false;
    }
    if (this->friction_roll != other.friction_roll) {
      return false;
    }
    return true;
  }
  bool operator!=(const GeomProperties_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GeomProperties_

// alias to use template instance with default allocator
using GeomProperties =
  mujoco_ros_msgs::msg::GeomProperties_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__GEOM_PROPERTIES__STRUCT_HPP_
