// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mujoco_ros_msgs:action/Step.idl
// generated code does not contain a copyright notice

#ifndef MUJOCO_ROS_MSGS__ACTION__DETAIL__STEP__STRUCT_H_
#define MUJOCO_ROS_MSGS__ACTION__DETAIL__STEP__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in action/Step in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__action__Step_Goal
{
  uint16_t num_steps;
} mujoco_ros_msgs__action__Step_Goal;

// Struct for a sequence of mujoco_ros_msgs__action__Step_Goal.
typedef struct mujoco_ros_msgs__action__Step_Goal__Sequence
{
  mujoco_ros_msgs__action__Step_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__action__Step_Goal__Sequence;


// Constants defined in the message

/// Struct defined in action/Step in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__action__Step_Result
{
  bool success;
} mujoco_ros_msgs__action__Step_Result;

// Struct for a sequence of mujoco_ros_msgs__action__Step_Result.
typedef struct mujoco_ros_msgs__action__Step_Result__Sequence
{
  mujoco_ros_msgs__action__Step_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__action__Step_Result__Sequence;


// Constants defined in the message

/// Struct defined in action/Step in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__action__Step_Feedback
{
  uint16_t steps_left;
} mujoco_ros_msgs__action__Step_Feedback;

// Struct for a sequence of mujoco_ros_msgs__action__Step_Feedback.
typedef struct mujoco_ros_msgs__action__Step_Feedback__Sequence
{
  mujoco_ros_msgs__action__Step_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__action__Step_Feedback__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "mujoco_ros_msgs/action/detail/step__struct.h"

/// Struct defined in action/Step in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__action__Step_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  mujoco_ros_msgs__action__Step_Goal goal;
} mujoco_ros_msgs__action__Step_SendGoal_Request;

// Struct for a sequence of mujoco_ros_msgs__action__Step_SendGoal_Request.
typedef struct mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence
{
  mujoco_ros_msgs__action__Step_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/Step in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__action__Step_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} mujoco_ros_msgs__action__Step_SendGoal_Response;

// Struct for a sequence of mujoco_ros_msgs__action__Step_SendGoal_Response.
typedef struct mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence
{
  mujoco_ros_msgs__action__Step_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/Step in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__action__Step_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} mujoco_ros_msgs__action__Step_GetResult_Request;

// Struct for a sequence of mujoco_ros_msgs__action__Step_GetResult_Request.
typedef struct mujoco_ros_msgs__action__Step_GetResult_Request__Sequence
{
  mujoco_ros_msgs__action__Step_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__action__Step_GetResult_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "mujoco_ros_msgs/action/detail/step__struct.h"

/// Struct defined in action/Step in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__action__Step_GetResult_Response
{
  int8_t status;
  mujoco_ros_msgs__action__Step_Result result;
} mujoco_ros_msgs__action__Step_GetResult_Response;

// Struct for a sequence of mujoco_ros_msgs__action__Step_GetResult_Response.
typedef struct mujoco_ros_msgs__action__Step_GetResult_Response__Sequence
{
  mujoco_ros_msgs__action__Step_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__action__Step_GetResult_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "mujoco_ros_msgs/action/detail/step__struct.h"

/// Struct defined in action/Step in the package mujoco_ros_msgs.
typedef struct mujoco_ros_msgs__action__Step_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  mujoco_ros_msgs__action__Step_Feedback feedback;
} mujoco_ros_msgs__action__Step_FeedbackMessage;

// Struct for a sequence of mujoco_ros_msgs__action__Step_FeedbackMessage.
typedef struct mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence
{
  mujoco_ros_msgs__action__Step_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_ROS_MSGS__ACTION__DETAIL__STEP__STRUCT_H_
