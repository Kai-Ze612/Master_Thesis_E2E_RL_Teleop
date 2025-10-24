// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mujoco_ros_msgs:action/Step.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__ACTION__DETAIL__STEP__BUILDER_HPP_
#define MUJOCO_ROS_MSGS__ACTION__DETAIL__STEP__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mujoco_ros_msgs/action/detail/step__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mujoco_ros_msgs
{

namespace action
{

namespace builder
{

class Init_Step_Goal_num_steps
{
public:
  Init_Step_Goal_num_steps()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::mujoco_ros_msgs::action::Step_Goal num_steps(::mujoco_ros_msgs::action::Step_Goal::_num_steps_type arg)
  {
    msg_.num_steps = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::action::Step_Goal>()
{
  return mujoco_ros_msgs::action::builder::Init_Step_Goal_num_steps();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace action
{

namespace builder
{

class Init_Step_Result_success
{
public:
  Init_Step_Result_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::mujoco_ros_msgs::action::Step_Result success(::mujoco_ros_msgs::action::Step_Result::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::action::Step_Result>()
{
  return mujoco_ros_msgs::action::builder::Init_Step_Result_success();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace action
{

namespace builder
{

class Init_Step_Feedback_steps_left
{
public:
  Init_Step_Feedback_steps_left()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::mujoco_ros_msgs::action::Step_Feedback steps_left(::mujoco_ros_msgs::action::Step_Feedback::_steps_left_type arg)
  {
    msg_.steps_left = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::action::Step_Feedback>()
{
  return mujoco_ros_msgs::action::builder::Init_Step_Feedback_steps_left();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace action
{

namespace builder
{

class Init_Step_SendGoal_Request_goal
{
public:
  explicit Init_Step_SendGoal_Request_goal(::mujoco_ros_msgs::action::Step_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::action::Step_SendGoal_Request goal(::mujoco_ros_msgs::action::Step_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_SendGoal_Request msg_;
};

class Init_Step_SendGoal_Request_goal_id
{
public:
  Init_Step_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Step_SendGoal_Request_goal goal_id(::mujoco_ros_msgs::action::Step_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_Step_SendGoal_Request_goal(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::action::Step_SendGoal_Request>()
{
  return mujoco_ros_msgs::action::builder::Init_Step_SendGoal_Request_goal_id();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace action
{

namespace builder
{

class Init_Step_SendGoal_Response_stamp
{
public:
  explicit Init_Step_SendGoal_Response_stamp(::mujoco_ros_msgs::action::Step_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::action::Step_SendGoal_Response stamp(::mujoco_ros_msgs::action::Step_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_SendGoal_Response msg_;
};

class Init_Step_SendGoal_Response_accepted
{
public:
  Init_Step_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Step_SendGoal_Response_stamp accepted(::mujoco_ros_msgs::action::Step_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_Step_SendGoal_Response_stamp(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::action::Step_SendGoal_Response>()
{
  return mujoco_ros_msgs::action::builder::Init_Step_SendGoal_Response_accepted();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace action
{

namespace builder
{

class Init_Step_GetResult_Request_goal_id
{
public:
  Init_Step_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::mujoco_ros_msgs::action::Step_GetResult_Request goal_id(::mujoco_ros_msgs::action::Step_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::action::Step_GetResult_Request>()
{
  return mujoco_ros_msgs::action::builder::Init_Step_GetResult_Request_goal_id();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace action
{

namespace builder
{

class Init_Step_GetResult_Response_result
{
public:
  explicit Init_Step_GetResult_Response_result(::mujoco_ros_msgs::action::Step_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::action::Step_GetResult_Response result(::mujoco_ros_msgs::action::Step_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_GetResult_Response msg_;
};

class Init_Step_GetResult_Response_status
{
public:
  Init_Step_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Step_GetResult_Response_result status(::mujoco_ros_msgs::action::Step_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_Step_GetResult_Response_result(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::action::Step_GetResult_Response>()
{
  return mujoco_ros_msgs::action::builder::Init_Step_GetResult_Response_status();
}

}  // namespace mujoco_ros_msgs


namespace mujoco_ros_msgs
{

namespace action
{

namespace builder
{

class Init_Step_FeedbackMessage_feedback
{
public:
  explicit Init_Step_FeedbackMessage_feedback(::mujoco_ros_msgs::action::Step_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::mujoco_ros_msgs::action::Step_FeedbackMessage feedback(::mujoco_ros_msgs::action::Step_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_FeedbackMessage msg_;
};

class Init_Step_FeedbackMessage_goal_id
{
public:
  Init_Step_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Step_FeedbackMessage_feedback goal_id(::mujoco_ros_msgs::action::Step_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_Step_FeedbackMessage_feedback(msg_);
  }

private:
  ::mujoco_ros_msgs::action::Step_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::mujoco_ros_msgs::action::Step_FeedbackMessage>()
{
  return mujoco_ros_msgs::action::builder::Init_Step_FeedbackMessage_goal_id();
}

}  // namespace mujoco_ros_msgs

#endif  // MUJOCO_ROS_MSGS__ACTION__DETAIL__STEP__BUILDER_HPP_
