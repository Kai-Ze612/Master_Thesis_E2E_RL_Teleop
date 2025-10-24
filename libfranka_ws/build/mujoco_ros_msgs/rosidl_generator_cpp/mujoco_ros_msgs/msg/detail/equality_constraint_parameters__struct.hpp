// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mujoco_ros_msgs:msg/EqualityConstraintParameters.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__STRUCT_HPP_
#define MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'type'
#include "mujoco_ros_msgs/msg/detail/equality_constraint_type__struct.hpp"
// Member 'solver_parameters'
#include "mujoco_ros_msgs/msg/detail/solver_parameters__struct.hpp"
// Member 'anchor'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"
// Member 'relpose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__mujoco_ros_msgs__msg__EqualityConstraintParameters __attribute__((deprecated))
#else
# define DEPRECATED__mujoco_ros_msgs__msg__EqualityConstraintParameters __declspec(deprecated)
#endif

namespace mujoco_ros_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct EqualityConstraintParameters_
{
  using Type = EqualityConstraintParameters_<ContainerAllocator>;

  explicit EqualityConstraintParameters_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : type(_init),
    solver_parameters(_init),
    anchor(_init),
    relpose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->name = "";
      this->active = false;
      this->class_param = "";
      this->element1 = "";
      this->element2 = "";
      this->torquescale = 0.0;
    }
  }

  explicit EqualityConstraintParameters_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : name(_alloc),
    type(_alloc, _init),
    solver_parameters(_alloc, _init),
    class_param(_alloc),
    element1(_alloc),
    element2(_alloc),
    anchor(_alloc, _init),
    relpose(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->name = "";
      this->active = false;
      this->class_param = "";
      this->element1 = "";
      this->element2 = "";
      this->torquescale = 0.0;
    }
  }

  // field types and members
  using _name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _name_type name;
  using _type_type =
    mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator>;
  _type_type type;
  using _solver_parameters_type =
    mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator>;
  _solver_parameters_type solver_parameters;
  using _active_type =
    bool;
  _active_type active;
  using _class_param_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _class_param_type class_param;
  using _element1_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _element1_type element1;
  using _element2_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _element2_type element2;
  using _torquescale_type =
    double;
  _torquescale_type torquescale;
  using _anchor_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _anchor_type anchor;
  using _relpose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _relpose_type relpose;
  using _polycoef_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _polycoef_type polycoef;

  // setters for named parameter idiom
  Type & set__name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->name = _arg;
    return *this;
  }
  Type & set__type(
    const mujoco_ros_msgs::msg::EqualityConstraintType_<ContainerAllocator> & _arg)
  {
    this->type = _arg;
    return *this;
  }
  Type & set__solver_parameters(
    const mujoco_ros_msgs::msg::SolverParameters_<ContainerAllocator> & _arg)
  {
    this->solver_parameters = _arg;
    return *this;
  }
  Type & set__active(
    const bool & _arg)
  {
    this->active = _arg;
    return *this;
  }
  Type & set__class_param(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->class_param = _arg;
    return *this;
  }
  Type & set__element1(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->element1 = _arg;
    return *this;
  }
  Type & set__element2(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->element2 = _arg;
    return *this;
  }
  Type & set__torquescale(
    const double & _arg)
  {
    this->torquescale = _arg;
    return *this;
  }
  Type & set__anchor(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->anchor = _arg;
    return *this;
  }
  Type & set__relpose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->relpose = _arg;
    return *this;
  }
  Type & set__polycoef(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->polycoef = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator> *;
  using ConstRawPtr =
    const mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mujoco_ros_msgs__msg__EqualityConstraintParameters
    std::shared_ptr<mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mujoco_ros_msgs__msg__EqualityConstraintParameters
    std::shared_ptr<mujoco_ros_msgs::msg::EqualityConstraintParameters_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const EqualityConstraintParameters_ & other) const
  {
    if (this->name != other.name) {
      return false;
    }
    if (this->type != other.type) {
      return false;
    }
    if (this->solver_parameters != other.solver_parameters) {
      return false;
    }
    if (this->active != other.active) {
      return false;
    }
    if (this->class_param != other.class_param) {
      return false;
    }
    if (this->element1 != other.element1) {
      return false;
    }
    if (this->element2 != other.element2) {
      return false;
    }
    if (this->torquescale != other.torquescale) {
      return false;
    }
    if (this->anchor != other.anchor) {
      return false;
    }
    if (this->relpose != other.relpose) {
      return false;
    }
    if (this->polycoef != other.polycoef) {
      return false;
    }
    return true;
  }
  bool operator!=(const EqualityConstraintParameters_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct EqualityConstraintParameters_

// alias to use template instance with default allocator
using EqualityConstraintParameters =
  mujoco_ros_msgs::msg::EqualityConstraintParameters_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__MSG__DETAIL__EQUALITY_CONSTRAINT_PARAMETERS__STRUCT_HPP_
