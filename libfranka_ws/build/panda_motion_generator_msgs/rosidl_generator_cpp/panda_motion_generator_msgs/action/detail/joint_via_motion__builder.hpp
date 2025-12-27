// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from panda_motion_generator_msgs:action/JointViaMotion.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__JOINT_VIA_MOTION__BUILDER_HPP_
#define PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__JOINT_VIA_MOTION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "panda_motion_generator_msgs/action/detail/joint_via_motion__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace panda_motion_generator_msgs
{

namespace action
{

namespace builder
{

class Init_JointViaMotion_Goal_v_scale
{
public:
  explicit Init_JointViaMotion_Goal_v_scale(::panda_motion_generator_msgs::action::JointViaMotion_Goal & msg)
  : msg_(msg)
  {}
  ::panda_motion_generator_msgs::action::JointViaMotion_Goal v_scale(::panda_motion_generator_msgs::action::JointViaMotion_Goal::_v_scale_type arg)
  {
    msg_.v_scale = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_Goal msg_;
};

class Init_JointViaMotion_Goal_via_poses
{
public:
  Init_JointViaMotion_Goal_via_poses()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointViaMotion_Goal_v_scale via_poses(::panda_motion_generator_msgs::action::JointViaMotion_Goal::_via_poses_type arg)
  {
    msg_.via_poses = std::move(arg);
    return Init_JointViaMotion_Goal_v_scale(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::action::JointViaMotion_Goal>()
{
  return panda_motion_generator_msgs::action::builder::Init_JointViaMotion_Goal_via_poses();
}

}  // namespace panda_motion_generator_msgs


namespace panda_motion_generator_msgs
{

namespace action
{

namespace builder
{

class Init_JointViaMotion_Result_result
{
public:
  Init_JointViaMotion_Result_result()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::panda_motion_generator_msgs::action::JointViaMotion_Result result(::panda_motion_generator_msgs::action::JointViaMotion_Result::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::action::JointViaMotion_Result>()
{
  return panda_motion_generator_msgs::action::builder::Init_JointViaMotion_Result_result();
}

}  // namespace panda_motion_generator_msgs


namespace panda_motion_generator_msgs
{

namespace action
{

namespace builder
{

class Init_JointViaMotion_Feedback_time_to_completion
{
public:
  explicit Init_JointViaMotion_Feedback_time_to_completion(::panda_motion_generator_msgs::action::JointViaMotion_Feedback & msg)
  : msg_(msg)
  {}
  ::panda_motion_generator_msgs::action::JointViaMotion_Feedback time_to_completion(::panda_motion_generator_msgs::action::JointViaMotion_Feedback::_time_to_completion_type arg)
  {
    msg_.time_to_completion = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_Feedback msg_;
};

class Init_JointViaMotion_Feedback_progress
{
public:
  Init_JointViaMotion_Feedback_progress()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointViaMotion_Feedback_time_to_completion progress(::panda_motion_generator_msgs::action::JointViaMotion_Feedback::_progress_type arg)
  {
    msg_.progress = std::move(arg);
    return Init_JointViaMotion_Feedback_time_to_completion(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::action::JointViaMotion_Feedback>()
{
  return panda_motion_generator_msgs::action::builder::Init_JointViaMotion_Feedback_progress();
}

}  // namespace panda_motion_generator_msgs


namespace panda_motion_generator_msgs
{

namespace action
{

namespace builder
{

class Init_JointViaMotion_SendGoal_Request_goal
{
public:
  explicit Init_JointViaMotion_SendGoal_Request_goal(::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Request goal(::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Request msg_;
};

class Init_JointViaMotion_SendGoal_Request_goal_id
{
public:
  Init_JointViaMotion_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointViaMotion_SendGoal_Request_goal goal_id(::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_JointViaMotion_SendGoal_Request_goal(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Request>()
{
  return panda_motion_generator_msgs::action::builder::Init_JointViaMotion_SendGoal_Request_goal_id();
}

}  // namespace panda_motion_generator_msgs


namespace panda_motion_generator_msgs
{

namespace action
{

namespace builder
{

class Init_JointViaMotion_SendGoal_Response_stamp
{
public:
  explicit Init_JointViaMotion_SendGoal_Response_stamp(::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Response stamp(::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Response msg_;
};

class Init_JointViaMotion_SendGoal_Response_accepted
{
public:
  Init_JointViaMotion_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointViaMotion_SendGoal_Response_stamp accepted(::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_JointViaMotion_SendGoal_Response_stamp(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::action::JointViaMotion_SendGoal_Response>()
{
  return panda_motion_generator_msgs::action::builder::Init_JointViaMotion_SendGoal_Response_accepted();
}

}  // namespace panda_motion_generator_msgs


namespace panda_motion_generator_msgs
{

namespace action
{

namespace builder
{

class Init_JointViaMotion_GetResult_Request_goal_id
{
public:
  Init_JointViaMotion_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Request goal_id(::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Request>()
{
  return panda_motion_generator_msgs::action::builder::Init_JointViaMotion_GetResult_Request_goal_id();
}

}  // namespace panda_motion_generator_msgs


namespace panda_motion_generator_msgs
{

namespace action
{

namespace builder
{

class Init_JointViaMotion_GetResult_Response_result
{
public:
  explicit Init_JointViaMotion_GetResult_Response_result(::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Response result(::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Response msg_;
};

class Init_JointViaMotion_GetResult_Response_status
{
public:
  Init_JointViaMotion_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointViaMotion_GetResult_Response_result status(::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_JointViaMotion_GetResult_Response_result(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::action::JointViaMotion_GetResult_Response>()
{
  return panda_motion_generator_msgs::action::builder::Init_JointViaMotion_GetResult_Response_status();
}

}  // namespace panda_motion_generator_msgs


namespace panda_motion_generator_msgs
{

namespace action
{

namespace builder
{

class Init_JointViaMotion_FeedbackMessage_feedback
{
public:
  explicit Init_JointViaMotion_FeedbackMessage_feedback(::panda_motion_generator_msgs::action::JointViaMotion_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::panda_motion_generator_msgs::action::JointViaMotion_FeedbackMessage feedback(::panda_motion_generator_msgs::action::JointViaMotion_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_FeedbackMessage msg_;
};

class Init_JointViaMotion_FeedbackMessage_goal_id
{
public:
  Init_JointViaMotion_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointViaMotion_FeedbackMessage_feedback goal_id(::panda_motion_generator_msgs::action::JointViaMotion_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_JointViaMotion_FeedbackMessage_feedback(msg_);
  }

private:
  ::panda_motion_generator_msgs::action::JointViaMotion_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::panda_motion_generator_msgs::action::JointViaMotion_FeedbackMessage>()
{
  return panda_motion_generator_msgs::action::builder::Init_JointViaMotion_FeedbackMessage_goal_id();
}

}  // namespace panda_motion_generator_msgs

#endif  // PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__JOINT_VIA_MOTION__BUILDER_HPP_
