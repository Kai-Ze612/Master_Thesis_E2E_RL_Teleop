// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from multi_mode_control_msgs:msg/ContactThresholds.idl
// generated code does not contain a copyright notice
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__rosidl_typesupport_fastrtps_cpp.hpp"
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__struct.hpp"

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
namespace multi_mode_control_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const multi_mode_control_msgs::msg::JointArray &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  multi_mode_control_msgs::msg::JointArray &);
size_t get_serialized_size(
  const multi_mode_control_msgs::msg::JointArray &,
  size_t current_alignment);
size_t
max_serialized_size_JointArray(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace multi_mode_control_msgs

// functions for multi_mode_control_msgs::msg::JointArray already declared above


namespace multi_mode_control_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_multi_mode_control_msgs
cdr_serialize(
  const multi_mode_control_msgs::msg::ContactThresholds & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: f_max
  {
    cdr << ros_message.f_max;
  }
  // Member: f_min
  {
    cdr << ros_message.f_min;
  }
  // Member: tau_max
  {
    size_t size = ros_message.tau_max.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.tau_max[i],
        cdr);
    }
  }
  // Member: tau_min
  {
    size_t size = ros_message.tau_min.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.tau_min[i],
        cdr);
    }
  }
  // Member: f_abs_max
  cdr << ros_message.f_abs_max;
  // Member: m_abs_max
  cdr << ros_message.m_abs_max;
  // Member: abs_mode
  cdr << (ros_message.abs_mode ? true : false);
  // Member: directional_mode
  cdr << (ros_message.directional_mode ? true : false);
  // Member: torque_mode
  cdr << (ros_message.torque_mode ? true : false);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_multi_mode_control_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  multi_mode_control_msgs::msg::ContactThresholds & ros_message)
{
  // Member: f_max
  {
    cdr >> ros_message.f_max;
  }

  // Member: f_min
  {
    cdr >> ros_message.f_min;
  }

  // Member: tau_max
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    ros_message.tau_max.resize(size);
    for (size_t i = 0; i < size; i++) {
      multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.tau_max[i]);
    }
  }

  // Member: tau_min
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    ros_message.tau_min.resize(size);
    for (size_t i = 0; i < size; i++) {
      multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.tau_min[i]);
    }
  }

  // Member: f_abs_max
  cdr >> ros_message.f_abs_max;

  // Member: m_abs_max
  cdr >> ros_message.m_abs_max;

  // Member: abs_mode
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.abs_mode = tmp ? true : false;
  }

  // Member: directional_mode
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.directional_mode = tmp ? true : false;
  }

  // Member: torque_mode
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.torque_mode = tmp ? true : false;
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_multi_mode_control_msgs
get_serialized_size(
  const multi_mode_control_msgs::msg::ContactThresholds & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: f_max
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_max[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_min
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_min[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: tau_max
  {
    size_t array_size = ros_message.tau_max.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.tau_max[index], current_alignment);
    }
  }
  // Member: tau_min
  {
    size_t array_size = ros_message.tau_min.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.tau_min[index], current_alignment);
    }
  }
  // Member: f_abs_max
  {
    size_t item_size = sizeof(ros_message.f_abs_max);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: m_abs_max
  {
    size_t item_size = sizeof(ros_message.m_abs_max);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: abs_mode
  {
    size_t item_size = sizeof(ros_message.abs_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: directional_mode
  {
    size_t item_size = sizeof(ros_message.directional_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: torque_mode
  {
    size_t item_size = sizeof(ros_message.torque_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_multi_mode_control_msgs
max_serialized_size_ContactThresholds(
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


  // Member: f_max
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_min
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: tau_max
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
      size_t inner_size =
        multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_JointArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: tau_min
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
      size_t inner_size =
        multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_JointArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: f_abs_max
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: m_abs_max
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: abs_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: directional_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: torque_mode
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
    using DataType = multi_mode_control_msgs::msg::ContactThresholds;
    is_plain =
      (
      offsetof(DataType, torque_mode) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _ContactThresholds__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const multi_mode_control_msgs::msg::ContactThresholds *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _ContactThresholds__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<multi_mode_control_msgs::msg::ContactThresholds *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _ContactThresholds__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const multi_mode_control_msgs::msg::ContactThresholds *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _ContactThresholds__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_ContactThresholds(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _ContactThresholds__callbacks = {
  "multi_mode_control_msgs::msg",
  "ContactThresholds",
  _ContactThresholds__cdr_serialize,
  _ContactThresholds__cdr_deserialize,
  _ContactThresholds__get_serialized_size,
  _ContactThresholds__max_serialized_size
};

static rosidl_message_type_support_t _ContactThresholds__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_ContactThresholds__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace multi_mode_control_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_multi_mode_control_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<multi_mode_control_msgs::msg::ContactThresholds>()
{
  return &multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::_ContactThresholds__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, multi_mode_control_msgs, msg, ContactThresholds)() {
  return &multi_mode_control_msgs::msg::typesupport_fastrtps_cpp::_ContactThresholds__handle;
}

#ifdef __cplusplus
}
#endif
