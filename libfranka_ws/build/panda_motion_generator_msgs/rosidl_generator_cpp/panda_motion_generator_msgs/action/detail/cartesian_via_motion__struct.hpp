// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from panda_motion_generator_msgs:action/CartesianViaMotion.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__CARTESIAN_VIA_MOTION__STRUCT_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__CARTESIAN_VIA_MOTION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'via_poses'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Goal __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Goal __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct CartesianViaMotion_Goal_
{
  using Type = CartesianViaMotion_Goal_<ContainerAllocator>;

  explicit CartesianViaMotion_Goal_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->v_scale = 0.0;
    }
  }

  explicit CartesianViaMotion_Goal_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->v_scale = 0.0;
    }
  }

  // field types and members
  using _via_poses_type =
    std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>>;
  _via_poses_type via_poses;
  using _v_scale_type =
    double;
  _v_scale_type v_scale;

  // setters for named parameter idiom
  Type & set__via_poses(
    const std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>> & _arg)
  {
    this->via_poses = _arg;
    return *this;
  }
  Type & set__v_scale(
    const double & _arg)
  {
    this->v_scale = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Goal
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Goal
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianViaMotion_Goal_ & other) const
  {
    if (this->via_poses != other.via_poses) {
      return false;
    }
    if (this->v_scale != other.v_scale) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianViaMotion_Goal_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianViaMotion_Goal_

// alias to use template instance with default allocator
using CartesianViaMotion_Goal =
  panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace panda_motion_generator_msgs


// Include directives for member types
// Member 'result'
#include "panda_motion_generator_msgs/msg/detail/simple_action_result__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Result __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Result __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct CartesianViaMotion_Result_
{
  using Type = CartesianViaMotion_Result_<ContainerAllocator>;

  explicit CartesianViaMotion_Result_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : result(_init)
  {
    (void)_init;
  }

  explicit CartesianViaMotion_Result_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : result(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _result_type =
    panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator>;
  _result_type result;

  // setters for named parameter idiom
  Type & set__result(
    const panda_motion_generator_msgs::msg::SimpleActionResult_<ContainerAllocator> & _arg)
  {
    this->result = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Result
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Result
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianViaMotion_Result_ & other) const
  {
    if (this->result != other.result) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianViaMotion_Result_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianViaMotion_Result_

// alias to use template instance with default allocator
using CartesianViaMotion_Result =
  panda_motion_generator_msgs::action::CartesianViaMotion_Result_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace panda_motion_generator_msgs


#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Feedback __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Feedback __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct CartesianViaMotion_Feedback_
{
  using Type = CartesianViaMotion_Feedback_<ContainerAllocator>;

  explicit CartesianViaMotion_Feedback_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->progress = 0.0f;
      this->time_to_completion = 0.0f;
    }
  }

  explicit CartesianViaMotion_Feedback_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->progress = 0.0f;
      this->time_to_completion = 0.0f;
    }
  }

  // field types and members
  using _progress_type =
    float;
  _progress_type progress;
  using _time_to_completion_type =
    float;
  _time_to_completion_type time_to_completion;

  // setters for named parameter idiom
  Type & set__progress(
    const float & _arg)
  {
    this->progress = _arg;
    return *this;
  }
  Type & set__time_to_completion(
    const float & _arg)
  {
    this->time_to_completion = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Feedback
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_Feedback
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianViaMotion_Feedback_ & other) const
  {
    if (this->progress != other.progress) {
      return false;
    }
    if (this->time_to_completion != other.time_to_completion) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianViaMotion_Feedback_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianViaMotion_Feedback_

// alias to use template instance with default allocator
using CartesianViaMotion_Feedback =
  panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace panda_motion_generator_msgs


// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"
// Member 'goal'
#include "panda_motion_generator_msgs/action/detail/cartesian_via_motion__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct CartesianViaMotion_SendGoal_Request_
{
  using Type = CartesianViaMotion_SendGoal_Request_<ContainerAllocator>;

  explicit CartesianViaMotion_SendGoal_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init),
    goal(_init)
  {
    (void)_init;
  }

  explicit CartesianViaMotion_SendGoal_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_alloc, _init),
    goal(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _goal_id_type =
    unique_identifier_msgs::msg::UUID_<ContainerAllocator>;
  _goal_id_type goal_id;
  using _goal_type =
    panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator>;
  _goal_type goal;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }
  Type & set__goal(
    const panda_motion_generator_msgs::action::CartesianViaMotion_Goal_<ContainerAllocator> & _arg)
  {
    this->goal = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianViaMotion_SendGoal_Request_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    if (this->goal != other.goal) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianViaMotion_SendGoal_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianViaMotion_SendGoal_Request_

// alias to use template instance with default allocator
using CartesianViaMotion_SendGoal_Request =
  panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace panda_motion_generator_msgs


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct CartesianViaMotion_SendGoal_Response_
{
  using Type = CartesianViaMotion_SendGoal_Response_<ContainerAllocator>;

  explicit CartesianViaMotion_SendGoal_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->accepted = false;
    }
  }

  explicit CartesianViaMotion_SendGoal_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->accepted = false;
    }
  }

  // field types and members
  using _accepted_type =
    bool;
  _accepted_type accepted;
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;

  // setters for named parameter idiom
  Type & set__accepted(
    const bool & _arg)
  {
    this->accepted = _arg;
    return *this;
  }
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianViaMotion_SendGoal_Response_ & other) const
  {
    if (this->accepted != other.accepted) {
      return false;
    }
    if (this->stamp != other.stamp) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianViaMotion_SendGoal_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianViaMotion_SendGoal_Response_

// alias to use template instance with default allocator
using CartesianViaMotion_SendGoal_Response =
  panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace panda_motion_generator_msgs

namespace panda_motion_generator_msgs
{

namespace action
{

struct CartesianViaMotion_SendGoal
{
  using Request = panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Request;
  using Response = panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal_Response;
};

}  // namespace action

}  // namespace panda_motion_generator_msgs


// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct CartesianViaMotion_GetResult_Request_
{
  using Type = CartesianViaMotion_GetResult_Request_<ContainerAllocator>;

  explicit CartesianViaMotion_GetResult_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init)
  {
    (void)_init;
  }

  explicit CartesianViaMotion_GetResult_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _goal_id_type =
    unique_identifier_msgs::msg::UUID_<ContainerAllocator>;
  _goal_id_type goal_id;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianViaMotion_GetResult_Request_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianViaMotion_GetResult_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianViaMotion_GetResult_Request_

// alias to use template instance with default allocator
using CartesianViaMotion_GetResult_Request =
  panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace panda_motion_generator_msgs


// Include directives for member types
// Member 'result'
// already included above
// #include "panda_motion_generator_msgs/action/detail/cartesian_via_motion__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct CartesianViaMotion_GetResult_Response_
{
  using Type = CartesianViaMotion_GetResult_Response_<ContainerAllocator>;

  explicit CartesianViaMotion_GetResult_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : result(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
    }
  }

  explicit CartesianViaMotion_GetResult_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : result(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
    }
  }

  // field types and members
  using _status_type =
    int8_t;
  _status_type status;
  using _result_type =
    panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator>;
  _result_type result;

  // setters for named parameter idiom
  Type & set__status(
    const int8_t & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__result(
    const panda_motion_generator_msgs::action::CartesianViaMotion_Result_<ContainerAllocator> & _arg)
  {
    this->result = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianViaMotion_GetResult_Response_ & other) const
  {
    if (this->status != other.status) {
      return false;
    }
    if (this->result != other.result) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianViaMotion_GetResult_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianViaMotion_GetResult_Response_

// alias to use template instance with default allocator
using CartesianViaMotion_GetResult_Response =
  panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace panda_motion_generator_msgs

namespace panda_motion_generator_msgs
{

namespace action
{

struct CartesianViaMotion_GetResult
{
  using Request = panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Request;
  using Response = panda_motion_generator_msgs::action::CartesianViaMotion_GetResult_Response;
};

}  // namespace action

}  // namespace panda_motion_generator_msgs


// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"
// Member 'feedback'
// already included above
// #include "panda_motion_generator_msgs/action/detail/cartesian_via_motion__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage __attribute__((deprecated))
#else
# define DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage __declspec(deprecated)
#endif

namespace panda_motion_generator_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct CartesianViaMotion_FeedbackMessage_
{
  using Type = CartesianViaMotion_FeedbackMessage_<ContainerAllocator>;

  explicit CartesianViaMotion_FeedbackMessage_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init),
    feedback(_init)
  {
    (void)_init;
  }

  explicit CartesianViaMotion_FeedbackMessage_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_alloc, _init),
    feedback(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _goal_id_type =
    unique_identifier_msgs::msg::UUID_<ContainerAllocator>;
  _goal_id_type goal_id;
  using _feedback_type =
    panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator>;
  _feedback_type feedback;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }
  Type & set__feedback(
    const panda_motion_generator_msgs::action::CartesianViaMotion_Feedback_<ContainerAllocator> & _arg)
  {
    this->feedback = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator> *;
  using ConstRawPtr =
    const panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage
    std::shared_ptr<panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianViaMotion_FeedbackMessage_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    if (this->feedback != other.feedback) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianViaMotion_FeedbackMessage_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianViaMotion_FeedbackMessage_

// alias to use template instance with default allocator
using CartesianViaMotion_FeedbackMessage =
  panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace panda_motion_generator_msgs

#include "action_msgs/srv/cancel_goal.hpp"
#include "action_msgs/msg/goal_info.hpp"
#include "action_msgs/msg/goal_status_array.hpp"

namespace panda_motion_generator_msgs
{

namespace action
{

struct CartesianViaMotion
{
  /// The goal message defined in the action definition.
  using Goal = panda_motion_generator_msgs::action::CartesianViaMotion_Goal;
  /// The result message defined in the action definition.
  using Result = panda_motion_generator_msgs::action::CartesianViaMotion_Result;
  /// The feedback message defined in the action definition.
  using Feedback = panda_motion_generator_msgs::action::CartesianViaMotion_Feedback;

  struct Impl
  {
    /// The send_goal service using a wrapped version of the goal message as a request.
    using SendGoalService = panda_motion_generator_msgs::action::CartesianViaMotion_SendGoal;
    /// The get_result service using a wrapped version of the result message as a response.
    using GetResultService = panda_motion_generator_msgs::action::CartesianViaMotion_GetResult;
    /// The feedback message with generic fields which wraps the feedback message.
    using FeedbackMessage = panda_motion_generator_msgs::action::CartesianViaMotion_FeedbackMessage;

    /// The generic service to cancel a goal.
    using CancelGoalService = action_msgs::srv::CancelGoal;
    /// The generic message for the status of a goal.
    using GoalStatusMessage = action_msgs::msg::GoalStatusArray;
  };
};

typedef struct CartesianViaMotion CartesianViaMotion;

}  // namespace action

}  // namespace panda_motion_generator_msgs

#endif  // PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__CARTESIAN_VIA_MOTION__STRUCT_HPP_
