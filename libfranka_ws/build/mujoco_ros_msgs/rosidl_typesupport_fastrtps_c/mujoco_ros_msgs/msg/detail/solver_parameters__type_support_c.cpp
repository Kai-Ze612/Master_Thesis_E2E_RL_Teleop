// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from mujoco_ros_msgs:msg/SolverParameters.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/solver_parameters__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "mujoco_ros_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "mujoco_ros_msgs/msg/detail/solver_parameters__struct.h"
#include "mujoco_ros_msgs/msg/detail/solver_parameters__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _SolverParameters__ros_msg_type = mujoco_ros_msgs__msg__SolverParameters;

static bool _SolverParameters__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _SolverParameters__ros_msg_type * ros_message = static_cast<const _SolverParameters__ros_msg_type *>(untyped_ros_message);
  // Field name: dmin
  {
    cdr << ros_message->dmin;
  }

  // Field name: dmax
  {
    cdr << ros_message->dmax;
  }

  // Field name: width
  {
    cdr << ros_message->width;
  }

  // Field name: midpoint
  {
    cdr << ros_message->midpoint;
  }

  // Field name: power
  {
    cdr << ros_message->power;
  }

  // Field name: timeconst
  {
    cdr << ros_message->timeconst;
  }

  // Field name: dampratio
  {
    cdr << ros_message->dampratio;
  }

  return true;
}

static bool _SolverParameters__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _SolverParameters__ros_msg_type * ros_message = static_cast<_SolverParameters__ros_msg_type *>(untyped_ros_message);
  // Field name: dmin
  {
    cdr >> ros_message->dmin;
  }

  // Field name: dmax
  {
    cdr >> ros_message->dmax;
  }

  // Field name: width
  {
    cdr >> ros_message->width;
  }

  // Field name: midpoint
  {
    cdr >> ros_message->midpoint;
  }

  // Field name: power
  {
    cdr >> ros_message->power;
  }

  // Field name: timeconst
  {
    cdr >> ros_message->timeconst;
  }

  // Field name: dampratio
  {
    cdr >> ros_message->dampratio;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mujoco_ros_msgs
size_t get_serialized_size_mujoco_ros_msgs__msg__SolverParameters(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _SolverParameters__ros_msg_type * ros_message = static_cast<const _SolverParameters__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name dmin
  {
    size_t item_size = sizeof(ros_message->dmin);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name dmax
  {
    size_t item_size = sizeof(ros_message->dmax);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name width
  {
    size_t item_size = sizeof(ros_message->width);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name midpoint
  {
    size_t item_size = sizeof(ros_message->midpoint);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name power
  {
    size_t item_size = sizeof(ros_message->power);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name timeconst
  {
    size_t item_size = sizeof(ros_message->timeconst);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name dampratio
  {
    size_t item_size = sizeof(ros_message->dampratio);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _SolverParameters__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_mujoco_ros_msgs__msg__SolverParameters(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mujoco_ros_msgs
size_t max_serialized_size_mujoco_ros_msgs__msg__SolverParameters(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: dmin
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: dmax
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: width
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: midpoint
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: power
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: timeconst
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: dampratio
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = mujoco_ros_msgs__msg__SolverParameters;
    is_plain =
      (
      offsetof(DataType, dampratio) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _SolverParameters__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_mujoco_ros_msgs__msg__SolverParameters(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_SolverParameters = {
  "mujoco_ros_msgs::msg",
  "SolverParameters",
  _SolverParameters__cdr_serialize,
  _SolverParameters__cdr_deserialize,
  _SolverParameters__get_serialized_size,
  _SolverParameters__max_serialized_size
};

static rosidl_message_type_support_t _SolverParameters__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_SolverParameters,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mujoco_ros_msgs, msg, SolverParameters)() {
  return &_SolverParameters__type_support;
}

#if defined(__cplusplus)
}
#endif
