// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from multi_mode_control_msgs:msg/ContactThresholds.idl
// generated code does not contain a copyright notice
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "multi_mode_control_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__struct.h"
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__functions.h"
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

#include "multi_mode_control_msgs/msg/detail/joint_array__functions.h"  // tau_max, tau_min

// forward declare type support functions
size_t get_serialized_size_multi_mode_control_msgs__msg__JointArray(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_multi_mode_control_msgs__msg__JointArray(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, multi_mode_control_msgs, msg, JointArray)();


using _ContactThresholds__ros_msg_type = multi_mode_control_msgs__msg__ContactThresholds;

static bool _ContactThresholds__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _ContactThresholds__ros_msg_type * ros_message = static_cast<const _ContactThresholds__ros_msg_type *>(untyped_ros_message);
  // Field name: f_max
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_max;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_min
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_min;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: tau_max
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, multi_mode_control_msgs, msg, JointArray
      )()->data);
    size_t size = ros_message->tau_max.size;
    auto array_ptr = ros_message->tau_max.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: tau_min
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, multi_mode_control_msgs, msg, JointArray
      )()->data);
    size_t size = ros_message->tau_min.size;
    auto array_ptr = ros_message->tau_min.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: f_abs_max
  {
    cdr << ros_message->f_abs_max;
  }

  // Field name: m_abs_max
  {
    cdr << ros_message->m_abs_max;
  }

  // Field name: abs_mode
  {
    cdr << (ros_message->abs_mode ? true : false);
  }

  // Field name: directional_mode
  {
    cdr << (ros_message->directional_mode ? true : false);
  }

  // Field name: torque_mode
  {
    cdr << (ros_message->torque_mode ? true : false);
  }

  return true;
}

static bool _ContactThresholds__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _ContactThresholds__ros_msg_type * ros_message = static_cast<_ContactThresholds__ros_msg_type *>(untyped_ros_message);
  // Field name: f_max
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_max;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_min
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_min;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: tau_max
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, multi_mode_control_msgs, msg, JointArray
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->tau_max.data) {
      multi_mode_control_msgs__msg__JointArray__Sequence__fini(&ros_message->tau_max);
    }
    if (!multi_mode_control_msgs__msg__JointArray__Sequence__init(&ros_message->tau_max, size)) {
      fprintf(stderr, "failed to create array for field 'tau_max'");
      return false;
    }
    auto array_ptr = ros_message->tau_max.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: tau_min
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, multi_mode_control_msgs, msg, JointArray
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->tau_min.data) {
      multi_mode_control_msgs__msg__JointArray__Sequence__fini(&ros_message->tau_min);
    }
    if (!multi_mode_control_msgs__msg__JointArray__Sequence__init(&ros_message->tau_min, size)) {
      fprintf(stderr, "failed to create array for field 'tau_min'");
      return false;
    }
    auto array_ptr = ros_message->tau_min.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: f_abs_max
  {
    cdr >> ros_message->f_abs_max;
  }

  // Field name: m_abs_max
  {
    cdr >> ros_message->m_abs_max;
  }

  // Field name: abs_mode
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->abs_mode = tmp ? true : false;
  }

  // Field name: directional_mode
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->directional_mode = tmp ? true : false;
  }

  // Field name: torque_mode
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->torque_mode = tmp ? true : false;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_multi_mode_control_msgs
size_t get_serialized_size_multi_mode_control_msgs__msg__ContactThresholds(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _ContactThresholds__ros_msg_type * ros_message = static_cast<const _ContactThresholds__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name f_max
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_max;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_min
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_min;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tau_max
  {
    size_t array_size = ros_message->tau_max.size;
    auto array_ptr = ros_message->tau_max.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_multi_mode_control_msgs__msg__JointArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name tau_min
  {
    size_t array_size = ros_message->tau_min.size;
    auto array_ptr = ros_message->tau_min.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_multi_mode_control_msgs__msg__JointArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name f_abs_max
  {
    size_t item_size = sizeof(ros_message->f_abs_max);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name m_abs_max
  {
    size_t item_size = sizeof(ros_message->m_abs_max);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name abs_mode
  {
    size_t item_size = sizeof(ros_message->abs_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name directional_mode
  {
    size_t item_size = sizeof(ros_message->directional_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name torque_mode
  {
    size_t item_size = sizeof(ros_message->torque_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _ContactThresholds__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_multi_mode_control_msgs__msg__ContactThresholds(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_multi_mode_control_msgs
size_t max_serialized_size_multi_mode_control_msgs__msg__ContactThresholds(
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

  // member: f_max
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_min
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: tau_max
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_multi_mode_control_msgs__msg__JointArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: tau_min
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_multi_mode_control_msgs__msg__JointArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: f_abs_max
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: m_abs_max
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: abs_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: directional_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: torque_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = multi_mode_control_msgs__msg__ContactThresholds;
    is_plain =
      (
      offsetof(DataType, torque_mode) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _ContactThresholds__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_multi_mode_control_msgs__msg__ContactThresholds(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_ContactThresholds = {
  "multi_mode_control_msgs::msg",
  "ContactThresholds",
  _ContactThresholds__cdr_serialize,
  _ContactThresholds__cdr_deserialize,
  _ContactThresholds__get_serialized_size,
  _ContactThresholds__max_serialized_size
};

static rosidl_message_type_support_t _ContactThresholds__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_ContactThresholds,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, multi_mode_control_msgs, msg, ContactThresholds)() {
  return &_ContactThresholds__type_support;
}

#if defined(__cplusplus)
}
#endif
