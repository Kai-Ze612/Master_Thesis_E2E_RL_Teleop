// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from mujoco_ros_msgs:msg/SimInfo.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/sim_info__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "mujoco_ros_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "mujoco_ros_msgs/msg/detail/sim_info__struct.h"
#include "mujoco_ros_msgs/msg/detail/sim_info__functions.h"
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

#include "mujoco_ros_msgs/msg/detail/state_uint__functions.h"  // loading_state
#include "rosidl_runtime_c/string.h"  // model_path
#include "rosidl_runtime_c/string_functions.h"  // model_path

// forward declare type support functions
size_t get_serialized_size_mujoco_ros_msgs__msg__StateUint(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_mujoco_ros_msgs__msg__StateUint(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mujoco_ros_msgs, msg, StateUint)();


using _SimInfo__ros_msg_type = mujoco_ros_msgs__msg__SimInfo;

static bool _SimInfo__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _SimInfo__ros_msg_type * ros_message = static_cast<const _SimInfo__ros_msg_type *>(untyped_ros_message);
  // Field name: model_path
  {
    const rosidl_runtime_c__String * str = &ros_message->model_path;
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

  // Field name: model_valid
  {
    cdr << (ros_message->model_valid ? true : false);
  }

  // Field name: load_count
  {
    cdr << ros_message->load_count;
  }

  // Field name: loading_state
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, mujoco_ros_msgs, msg, StateUint
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->loading_state, cdr))
    {
      return false;
    }
  }

  // Field name: paused
  {
    cdr << (ros_message->paused ? true : false);
  }

  // Field name: pending_sim_steps
  {
    cdr << ros_message->pending_sim_steps;
  }

  // Field name: rt_measured
  {
    cdr << ros_message->rt_measured;
  }

  // Field name: rt_setting
  {
    cdr << ros_message->rt_setting;
  }

  return true;
}

static bool _SimInfo__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _SimInfo__ros_msg_type * ros_message = static_cast<_SimInfo__ros_msg_type *>(untyped_ros_message);
  // Field name: model_path
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->model_path.data) {
      rosidl_runtime_c__String__init(&ros_message->model_path);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->model_path,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'model_path'\n");
      return false;
    }
  }

  // Field name: model_valid
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->model_valid = tmp ? true : false;
  }

  // Field name: load_count
  {
    cdr >> ros_message->load_count;
  }

  // Field name: loading_state
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, mujoco_ros_msgs, msg, StateUint
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->loading_state))
    {
      return false;
    }
  }

  // Field name: paused
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->paused = tmp ? true : false;
  }

  // Field name: pending_sim_steps
  {
    cdr >> ros_message->pending_sim_steps;
  }

  // Field name: rt_measured
  {
    cdr >> ros_message->rt_measured;
  }

  // Field name: rt_setting
  {
    cdr >> ros_message->rt_setting;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mujoco_ros_msgs
size_t get_serialized_size_mujoco_ros_msgs__msg__SimInfo(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _SimInfo__ros_msg_type * ros_message = static_cast<const _SimInfo__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name model_path
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->model_path.size + 1);
  // field.name model_valid
  {
    size_t item_size = sizeof(ros_message->model_valid);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name load_count
  {
    size_t item_size = sizeof(ros_message->load_count);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name loading_state

  current_alignment += get_serialized_size_mujoco_ros_msgs__msg__StateUint(
    &(ros_message->loading_state), current_alignment);
  // field.name paused
  {
    size_t item_size = sizeof(ros_message->paused);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name pending_sim_steps
  {
    size_t item_size = sizeof(ros_message->pending_sim_steps);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name rt_measured
  {
    size_t item_size = sizeof(ros_message->rt_measured);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name rt_setting
  {
    size_t item_size = sizeof(ros_message->rt_setting);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _SimInfo__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_mujoco_ros_msgs__msg__SimInfo(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mujoco_ros_msgs
size_t max_serialized_size_mujoco_ros_msgs__msg__SimInfo(
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

  // member: model_path
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
  // member: model_valid
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: load_count
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }
  // member: loading_state
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_mujoco_ros_msgs__msg__StateUint(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: paused
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: pending_sim_steps
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }
  // member: rt_measured
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: rt_setting
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
    using DataType = mujoco_ros_msgs__msg__SimInfo;
    is_plain =
      (
      offsetof(DataType, rt_setting) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _SimInfo__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_mujoco_ros_msgs__msg__SimInfo(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_SimInfo = {
  "mujoco_ros_msgs::msg",
  "SimInfo",
  _SimInfo__cdr_serialize,
  _SimInfo__cdr_deserialize,
  _SimInfo__get_serialized_size,
  _SimInfo__max_serialized_size
};

static rosidl_message_type_support_t _SimInfo__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_SimInfo,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mujoco_ros_msgs, msg, SimInfo)() {
  return &_SimInfo__type_support;
}

#if defined(__cplusplus)
}
#endif
