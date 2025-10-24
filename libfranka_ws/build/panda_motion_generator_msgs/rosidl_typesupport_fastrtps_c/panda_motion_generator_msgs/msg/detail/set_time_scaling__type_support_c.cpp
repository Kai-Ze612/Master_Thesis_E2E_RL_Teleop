// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from panda_motion_generator_msgs:msg/SetTimeScaling.idl
// generated code does not contain a copyright notice
#include "panda_motion_generator_msgs/msg/detail/set_time_scaling__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "panda_motion_generator_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "panda_motion_generator_msgs/msg/detail/set_time_scaling__struct.h"
#include "panda_motion_generator_msgs/msg/detail/set_time_scaling__functions.h"
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


using _SetTimeScaling__ros_msg_type = panda_motion_generator_msgs__msg__SetTimeScaling;

static bool _SetTimeScaling__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _SetTimeScaling__ros_msg_type * ros_message = static_cast<const _SetTimeScaling__ros_msg_type *>(untyped_ros_message);
  // Field name: steps
  {
    cdr << ros_message->steps;
  }

  // Field name: scaling
  {
    cdr << ros_message->scaling;
  }

  return true;
}

static bool _SetTimeScaling__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _SetTimeScaling__ros_msg_type * ros_message = static_cast<_SetTimeScaling__ros_msg_type *>(untyped_ros_message);
  // Field name: steps
  {
    cdr >> ros_message->steps;
  }

  // Field name: scaling
  {
    cdr >> ros_message->scaling;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_panda_motion_generator_msgs
size_t get_serialized_size_panda_motion_generator_msgs__msg__SetTimeScaling(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _SetTimeScaling__ros_msg_type * ros_message = static_cast<const _SetTimeScaling__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name steps
  {
    size_t item_size = sizeof(ros_message->steps);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name scaling
  {
    size_t item_size = sizeof(ros_message->scaling);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _SetTimeScaling__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_panda_motion_generator_msgs__msg__SetTimeScaling(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_panda_motion_generator_msgs
size_t max_serialized_size_panda_motion_generator_msgs__msg__SetTimeScaling(
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

  // member: steps
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: scaling
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
    using DataType = panda_motion_generator_msgs__msg__SetTimeScaling;
    is_plain =
      (
      offsetof(DataType, scaling) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _SetTimeScaling__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_panda_motion_generator_msgs__msg__SetTimeScaling(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_SetTimeScaling = {
  "panda_motion_generator_msgs::msg",
  "SetTimeScaling",
  _SetTimeScaling__cdr_serialize,
  _SetTimeScaling__cdr_deserialize,
  _SetTimeScaling__get_serialized_size,
  _SetTimeScaling__max_serialized_size
};

static rosidl_message_type_support_t _SetTimeScaling__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_SetTimeScaling,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, panda_motion_generator_msgs, msg, SetTimeScaling)() {
  return &_SetTimeScaling__type_support;
}

#if defined(__cplusplus)
}
#endif
