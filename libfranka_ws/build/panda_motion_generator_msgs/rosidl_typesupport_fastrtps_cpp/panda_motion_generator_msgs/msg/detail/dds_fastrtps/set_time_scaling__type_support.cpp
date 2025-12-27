// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from panda_motion_generator_msgs:msg/SetTimeScaling.idl
// generated code does not contain a copyright notice
#include "panda_motion_generator_msgs/msg/detail/set_time_scaling__rosidl_typesupport_fastrtps_cpp.hpp"
#include "panda_motion_generator_msgs/msg/detail/set_time_scaling__struct.hpp"

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

namespace panda_motion_generator_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_panda_motion_generator_msgs
cdr_serialize(
  const panda_motion_generator_msgs::msg::SetTimeScaling & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: steps
  cdr << ros_message.steps;
  // Member: scaling
  cdr << ros_message.scaling;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_panda_motion_generator_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  panda_motion_generator_msgs::msg::SetTimeScaling & ros_message)
{
  // Member: steps
  cdr >> ros_message.steps;

  // Member: scaling
  cdr >> ros_message.scaling;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_panda_motion_generator_msgs
get_serialized_size(
  const panda_motion_generator_msgs::msg::SetTimeScaling & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: steps
  {
    size_t item_size = sizeof(ros_message.steps);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: scaling
  {
    size_t item_size = sizeof(ros_message.scaling);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_panda_motion_generator_msgs
max_serialized_size_SetTimeScaling(
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


  // Member: steps
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: scaling
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
    using DataType = panda_motion_generator_msgs::msg::SetTimeScaling;
    is_plain =
      (
      offsetof(DataType, scaling) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _SetTimeScaling__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const panda_motion_generator_msgs::msg::SetTimeScaling *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _SetTimeScaling__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<panda_motion_generator_msgs::msg::SetTimeScaling *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _SetTimeScaling__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const panda_motion_generator_msgs::msg::SetTimeScaling *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _SetTimeScaling__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_SetTimeScaling(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _SetTimeScaling__callbacks = {
  "panda_motion_generator_msgs::msg",
  "SetTimeScaling",
  _SetTimeScaling__cdr_serialize,
  _SetTimeScaling__cdr_deserialize,
  _SetTimeScaling__get_serialized_size,
  _SetTimeScaling__max_serialized_size
};

static rosidl_message_type_support_t _SetTimeScaling__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_SetTimeScaling__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace panda_motion_generator_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_panda_motion_generator_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<panda_motion_generator_msgs::msg::SetTimeScaling>()
{
  return &panda_motion_generator_msgs::msg::typesupport_fastrtps_cpp::_SetTimeScaling__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, panda_motion_generator_msgs, msg, SetTimeScaling)() {
  return &panda_motion_generator_msgs::msg::typesupport_fastrtps_cpp::_SetTimeScaling__handle;
}

#ifdef __cplusplus
}
#endif
