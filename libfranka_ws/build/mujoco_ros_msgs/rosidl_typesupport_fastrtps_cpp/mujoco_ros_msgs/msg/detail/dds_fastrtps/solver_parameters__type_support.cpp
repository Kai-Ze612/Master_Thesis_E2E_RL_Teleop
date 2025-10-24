// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from mujoco_ros_msgs:msg/SolverParameters.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/solver_parameters__rosidl_typesupport_fastrtps_cpp.hpp"
#include "mujoco_ros_msgs/msg/detail/solver_parameters__struct.hpp"

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
  const mujoco_ros_msgs::msg::SolverParameters & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: dmin
  cdr << ros_message.dmin;
  // Member: dmax
  cdr << ros_message.dmax;
  // Member: width
  cdr << ros_message.width;
  // Member: midpoint
  cdr << ros_message.midpoint;
  // Member: power
  cdr << ros_message.power;
  // Member: timeconst
  cdr << ros_message.timeconst;
  // Member: dampratio
  cdr << ros_message.dampratio;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mujoco_ros_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  mujoco_ros_msgs::msg::SolverParameters & ros_message)
{
  // Member: dmin
  cdr >> ros_message.dmin;

  // Member: dmax
  cdr >> ros_message.dmax;

  // Member: width
  cdr >> ros_message.width;

  // Member: midpoint
  cdr >> ros_message.midpoint;

  // Member: power
  cdr >> ros_message.power;

  // Member: timeconst
  cdr >> ros_message.timeconst;

  // Member: dampratio
  cdr >> ros_message.dampratio;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mujoco_ros_msgs
get_serialized_size(
  const mujoco_ros_msgs::msg::SolverParameters & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: dmin
  {
    size_t item_size = sizeof(ros_message.dmin);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: dmax
  {
    size_t item_size = sizeof(ros_message.dmax);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: width
  {
    size_t item_size = sizeof(ros_message.width);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: midpoint
  {
    size_t item_size = sizeof(ros_message.midpoint);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: power
  {
    size_t item_size = sizeof(ros_message.power);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: timeconst
  {
    size_t item_size = sizeof(ros_message.timeconst);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: dampratio
  {
    size_t item_size = sizeof(ros_message.dampratio);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mujoco_ros_msgs
max_serialized_size_SolverParameters(
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


  // Member: dmin
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: dmax
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: width
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: midpoint
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: power
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: timeconst
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: dampratio
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
    using DataType = mujoco_ros_msgs::msg::SolverParameters;
    is_plain =
      (
      offsetof(DataType, dampratio) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _SolverParameters__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const mujoco_ros_msgs::msg::SolverParameters *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _SolverParameters__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<mujoco_ros_msgs::msg::SolverParameters *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _SolverParameters__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const mujoco_ros_msgs::msg::SolverParameters *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _SolverParameters__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_SolverParameters(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _SolverParameters__callbacks = {
  "mujoco_ros_msgs::msg",
  "SolverParameters",
  _SolverParameters__cdr_serialize,
  _SolverParameters__cdr_deserialize,
  _SolverParameters__get_serialized_size,
  _SolverParameters__max_serialized_size
};

static rosidl_message_type_support_t _SolverParameters__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_SolverParameters__callbacks,
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
get_message_type_support_handle<mujoco_ros_msgs::msg::SolverParameters>()
{
  return &mujoco_ros_msgs::msg::typesupport_fastrtps_cpp::_SolverParameters__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, mujoco_ros_msgs, msg, SolverParameters)() {
  return &mujoco_ros_msgs::msg::typesupport_fastrtps_cpp::_SolverParameters__handle;
}

#ifdef __cplusplus
}
#endif
