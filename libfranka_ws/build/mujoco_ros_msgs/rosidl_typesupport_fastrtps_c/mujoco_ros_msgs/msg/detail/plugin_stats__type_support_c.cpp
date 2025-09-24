// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from mujoco_ros_msgs:msg/PluginStats.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/plugin_stats__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "mujoco_ros_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "mujoco_ros_msgs/msg/detail/plugin_stats__struct.h"
#include "mujoco_ros_msgs/msg/detail/plugin_stats__functions.h"
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

#include "rosidl_runtime_c/string.h"  // plugin_type
#include "rosidl_runtime_c/string_functions.h"  // plugin_type

// forward declare type support functions


using _PluginStats__ros_msg_type = mujoco_ros_msgs__msg__PluginStats;

static bool _PluginStats__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _PluginStats__ros_msg_type * ros_message = static_cast<const _PluginStats__ros_msg_type *>(untyped_ros_message);
  // Field name: plugin_type
  {
    const rosidl_runtime_c__String * str = &ros_message->plugin_type;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: load_time
  {
    cdr << ros_message->load_time;
  }

  // Field name: reset_time
  {
    cdr << ros_message->reset_time;
  }

  // Field name: ema_steptime_control
  {
    cdr << ros_message->ema_steptime_control;
  }

  // Field name: ema_steptime_passive
  {
    cdr << ros_message->ema_steptime_passive;
  }

  // Field name: ema_steptime_render
  {
    cdr << ros_message->ema_steptime_render;
  }

  // Field name: ema_steptime_last_stage
  {
    cdr << ros_message->ema_steptime_last_stage;
  }

  return true;
}

static bool _PluginStats__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _PluginStats__ros_msg_type * ros_message = static_cast<_PluginStats__ros_msg_type *>(untyped_ros_message);
  // Field name: plugin_type
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->plugin_type.data) {
      rosidl_runtime_c__String__init(&ros_message->plugin_type);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->plugin_type,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'plugin_type'\n");
      return false;
    }
  }

  // Field name: load_time
  {
    cdr >> ros_message->load_time;
  }

  // Field name: reset_time
  {
    cdr >> ros_message->reset_time;
  }

  // Field name: ema_steptime_control
  {
    cdr >> ros_message->ema_steptime_control;
  }

  // Field name: ema_steptime_passive
  {
    cdr >> ros_message->ema_steptime_passive;
  }

  // Field name: ema_steptime_render
  {
    cdr >> ros_message->ema_steptime_render;
  }

  // Field name: ema_steptime_last_stage
  {
    cdr >> ros_message->ema_steptime_last_stage;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mujoco_ros_msgs
size_t get_serialized_size_mujoco_ros_msgs__msg__PluginStats(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PluginStats__ros_msg_type * ros_message = static_cast<const _PluginStats__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name plugin_type
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->plugin_type.size + 1);
  // field.name load_time
  {
    size_t item_size = sizeof(ros_message->load_time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name reset_time
  {
    size_t item_size = sizeof(ros_message->reset_time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ema_steptime_control
  {
    size_t item_size = sizeof(ros_message->ema_steptime_control);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ema_steptime_passive
  {
    size_t item_size = sizeof(ros_message->ema_steptime_passive);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ema_steptime_render
  {
    size_t item_size = sizeof(ros_message->ema_steptime_render);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ema_steptime_last_stage
  {
    size_t item_size = sizeof(ros_message->ema_steptime_last_stage);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _PluginStats__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_mujoco_ros_msgs__msg__PluginStats(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mujoco_ros_msgs
size_t max_serialized_size_mujoco_ros_msgs__msg__PluginStats(
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

  // member: plugin_type
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: load_time
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: reset_time
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: ema_steptime_control
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: ema_steptime_passive
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: ema_steptime_render
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: ema_steptime_last_stage
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = mujoco_ros_msgs__msg__PluginStats;
    is_plain =
      (
      offsetof(DataType, ema_steptime_last_stage) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _PluginStats__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_mujoco_ros_msgs__msg__PluginStats(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_PluginStats = {
  "mujoco_ros_msgs::msg",
  "PluginStats",
  _PluginStats__cdr_serialize,
  _PluginStats__cdr_deserialize,
  _PluginStats__get_serialized_size,
  _PluginStats__max_serialized_size
};

static rosidl_message_type_support_t _PluginStats__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PluginStats,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mujoco_ros_msgs, msg, PluginStats)() {
  return &_PluginStats__type_support;
}

#if defined(__cplusplus)
}
#endif
