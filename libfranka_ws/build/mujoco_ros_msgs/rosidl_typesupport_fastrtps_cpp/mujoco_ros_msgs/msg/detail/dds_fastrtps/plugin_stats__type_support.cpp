// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from mujoco_ros_msgs:msg/PluginStats.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/plugin_stats__rosidl_typesupport_fastrtps_cpp.hpp"
#include "mujoco_ros_msgs/msg/detail/plugin_stats__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace mujoco_ros_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mujoco_ros_msgs
cdr_serialize(
  const mujoco_ros_msgs::msg::PluginStats & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: plugin_type
  cdr << ros_message.plugin_type;
  // Member: load_time
  cdr << ros_message.load_time;
  // Member: reset_time
  cdr << ros_message.reset_time;
  // Member: ema_steptime_control
  cdr << ros_message.ema_steptime_control;
  // Member: ema_steptime_passive
  cdr << ros_message.ema_steptime_passive;
  // Member: ema_steptime_render
  cdr << ros_message.ema_steptime_render;
  // Member: ema_steptime_last_stage
  cdr << ros_message.ema_steptime_last_stage;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mujoco_ros_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  mujoco_ros_msgs::msg::PluginStats & ros_message)
{
  // Member: plugin_type
  cdr >> ros_message.plugin_type;

  // Member: load_time
  cdr >> ros_message.load_time;

  // Member: reset_time
  cdr >> ros_message.reset_time;

  // Member: ema_steptime_control
  cdr >> ros_message.ema_steptime_control;

  // Member: ema_steptime_passive
  cdr >> ros_message.ema_steptime_passive;

  // Member: ema_steptime_render
  cdr >> ros_message.ema_steptime_render;

  // Member: ema_steptime_last_stage
  cdr >> ros_message.ema_steptime_last_stage;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mujoco_ros_msgs
get_serialized_size(
  const mujoco_ros_msgs::msg::PluginStats & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: plugin_type
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.plugin_type.size() + 1);
  // Member: load_time
  {
    size_t item_size = sizeof(ros_message.load_time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: reset_time
  {
    size_t item_size = sizeof(ros_message.reset_time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ema_steptime_control
  {
    size_t item_size = sizeof(ros_message.ema_steptime_control);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ema_steptime_passive
  {
    size_t item_size = sizeof(ros_message.ema_steptime_passive);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ema_steptime_render
  {
    size_t item_size = sizeof(ros_message.ema_steptime_render);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ema_steptime_last_stage
  {
    size_t item_size = sizeof(ros_message.ema_steptime_last_stage);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mujoco_ros_msgs
max_serialized_size_PluginStats(
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


  // Member: plugin_type
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

  // Member: load_time
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: reset_time
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: ema_steptime_control
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: ema_steptime_passive
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: ema_steptime_render
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: ema_steptime_last_stage
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
    using DataType = mujoco_ros_msgs::msg::PluginStats;
    is_plain =
      (
      offsetof(DataType, ema_steptime_last_stage) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _PluginStats__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const mujoco_ros_msgs::msg::PluginStats *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _PluginStats__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<mujoco_ros_msgs::msg::PluginStats *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _PluginStats__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const mujoco_ros_msgs::msg::PluginStats *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _PluginStats__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_PluginStats(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _PluginStats__callbacks = {
  "mujoco_ros_msgs::msg",
  "PluginStats",
  _PluginStats__cdr_serialize,
  _PluginStats__cdr_deserialize,
  _PluginStats__get_serialized_size,
  _PluginStats__max_serialized_size
};

static rosidl_message_type_support_t _PluginStats__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_PluginStats__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace mujoco_ros_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_mujoco_ros_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<mujoco_ros_msgs::msg::PluginStats>()
{
  return &mujoco_ros_msgs::msg::typesupport_fastrtps_cpp::_PluginStats__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, mujoco_ros_msgs, msg, PluginStats)() {
  return &mujoco_ros_msgs::msg::typesupport_fastrtps_cpp::_PluginStats__handle;
}

#ifdef __cplusplus
}
#endif
