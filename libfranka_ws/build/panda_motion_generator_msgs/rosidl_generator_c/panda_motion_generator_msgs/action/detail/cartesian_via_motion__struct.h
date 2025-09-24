// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from panda_motion_generator_msgs:action/CartesianViaMotion.idl
// generated code does not contain a copyright notice

#ifndef PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__CARTESIAN_VIA_MOTION__STRUCT_H_
#define PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__CARTESIAN_VIA_MOTION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'via_poses'
#include "geometry_msgs/msg/detail/pose__struct.h"

/// Struct defined in action/CartesianViaMotion in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_Goal
{
  /// orientation zero will keep current orientation
  geometry_msgs__msg__Pose__Sequence via_poses;
  /// (0-1] of maximum velocity to be used for motion
  double v_scale;
} panda_motion_generator_msgs__action__CartesianViaMotion_Goal;

// Struct for a sequence of panda_motion_generator_msgs__action__CartesianViaMotion_Goal.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_Goal__Sequence
{
  panda_motion_generator_msgs__action__CartesianViaMotion_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__action__CartesianViaMotion_Goal__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
#include "panda_motion_generator_msgs/msg/detail/simple_action_result__struct.h"

/// Struct defined in action/CartesianViaMotion in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_Result
{
  panda_motion_generator_msgs__msg__SimpleActionResult result;
} panda_motion_generator_msgs__action__CartesianViaMotion_Result;

// Struct for a sequence of panda_motion_generator_msgs__action__CartesianViaMotion_Result.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_Result__Sequence
{
  panda_motion_generator_msgs__action__CartesianViaMotion_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__action__CartesianViaMotion_Result__Sequence;


// Constants defined in the message

/// Struct defined in action/CartesianViaMotion in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_Feedback
{
  float progress;
  float time_to_completion;
} panda_motion_generator_msgs__action__CartesianViaMotion_Feedback;

// Struct for a sequence of panda_motion_generator_msgs__action__CartesianViaMotion_Feedback.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_Feedback__Sequence
{
  panda_motion_generator_msgs__action__CartesianViaMotion_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__action__CartesianViaMotion_Feedback__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "panda_motion_generator_msgs/action/detail/cartesian_via_motion__struct.h"

/// Struct defined in action/CartesianViaMotion in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  panda_motion_generator_msgs__action__CartesianViaMotion_Goal goal;
} panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request;

// Struct for a sequence of panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request__Sequence
{
  panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/CartesianViaMotion in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response;

// Struct for a sequence of panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response__Sequence
{
  panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__action__CartesianViaMotion_SendGoal_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/CartesianViaMotion in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request;

// Struct for a sequence of panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request__Sequence
{
  panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "panda_motion_generator_msgs/action/detail/cartesian_via_motion__struct.h"

/// Struct defined in action/CartesianViaMotion in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response
{
  int8_t status;
  panda_motion_generator_msgs__action__CartesianViaMotion_Result result;
} panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response;

// Struct for a sequence of panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response__Sequence
{
  panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__action__CartesianViaMotion_GetResult_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "panda_motion_generator_msgs/action/detail/cartesian_via_motion__struct.h"

/// Struct defined in action/CartesianViaMotion in the package panda_motion_generator_msgs.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  panda_motion_generator_msgs__action__CartesianViaMotion_Feedback feedback;
} panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage;

// Struct for a sequence of panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage.
typedef struct panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage__Sequence
{
  panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} panda_motion_generator_msgs__action__CartesianViaMotion_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PANDA_MOTION_GENERATOR_MSGS__ACTION__DETAIL__CARTESIAN_VIA_MOTION__STRUCT_H_
