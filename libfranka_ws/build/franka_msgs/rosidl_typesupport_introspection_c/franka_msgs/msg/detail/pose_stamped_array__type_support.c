// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from franka_msgs:msg/PoseStampedArray.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "franka_msgs/msg/detail/pose_stamped_array__rosidl_typesupport_introspection_c.h"
#include "franka_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "franka_msgs/msg/detail/pose_stamped_array__functions.h"
#include "franka_msgs/msg/detail/pose_stamped_array__struct.h"


// Include directives for member types
// Member `stampedposes`
#include "geometry_msgs/msg/pose_stamped.h"
// Member `stampedposes`
#include "geometry_msgs/msg/detail/pose_stamped__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  franka_msgs__msg__PoseStampedArray__init(message_memory);
}

void franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_fini_function(void * message_memory)
{
  franka_msgs__msg__PoseStampedArray__fini(message_memory);
}

size_t franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__size_function__PoseStampedArray__stampedposes(
  const void * untyped_member)
{
  const geometry_msgs__msg__PoseStamped__Sequence * member =
    (const geometry_msgs__msg__PoseStamped__Sequence *)(untyped_member);
  return member->size;
}

const void * franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__get_const_function__PoseStampedArray__stampedposes(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__PoseStamped__Sequence * member =
    (const geometry_msgs__msg__PoseStamped__Sequence *)(untyped_member);
  return &member->data[index];
}

void * franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__get_function__PoseStampedArray__stampedposes(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__PoseStamped__Sequence * member =
    (geometry_msgs__msg__PoseStamped__Sequence *)(untyped_member);
  return &member->data[index];
}

void franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__fetch_function__PoseStampedArray__stampedposes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__PoseStamped * item =
    ((const geometry_msgs__msg__PoseStamped *)
    franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__get_const_function__PoseStampedArray__stampedposes(untyped_member, index));
  geometry_msgs__msg__PoseStamped * value =
    (geometry_msgs__msg__PoseStamped *)(untyped_value);
  *value = *item;
}

void franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__assign_function__PoseStampedArray__stampedposes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__PoseStamped * item =
    ((geometry_msgs__msg__PoseStamped *)
    franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__get_function__PoseStampedArray__stampedposes(untyped_member, index));
  const geometry_msgs__msg__PoseStamped * value =
    (const geometry_msgs__msg__PoseStamped *)(untyped_value);
  *item = *value;
}

bool franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__resize_function__PoseStampedArray__stampedposes(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__PoseStamped__Sequence * member =
    (geometry_msgs__msg__PoseStamped__Sequence *)(untyped_member);
  geometry_msgs__msg__PoseStamped__Sequence__fini(member);
  return geometry_msgs__msg__PoseStamped__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_message_member_array[1] = {
  {
    "stampedposes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(franka_msgs__msg__PoseStampedArray, stampedposes),  // bytes offset in struct
    NULL,  // default value
    franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__size_function__PoseStampedArray__stampedposes,  // size() function pointer
    franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__get_const_function__PoseStampedArray__stampedposes,  // get_const(index) function pointer
    franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__get_function__PoseStampedArray__stampedposes,  // get(index) function pointer
    franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__fetch_function__PoseStampedArray__stampedposes,  // fetch(index, &value) function pointer
    franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__assign_function__PoseStampedArray__stampedposes,  // assign(index, value) function pointer
    franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__resize_function__PoseStampedArray__stampedposes  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_message_members = {
  "franka_msgs__msg",  // message namespace
  "PoseStampedArray",  // message name
  1,  // number of fields
  sizeof(franka_msgs__msg__PoseStampedArray),
  franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_message_member_array,  // message members
  franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_init_function,  // function to initialize message memory (memory has to be allocated)
  franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_message_type_support_handle = {
  0,
  &franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_franka_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, franka_msgs, msg, PoseStampedArray)() {
  franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PoseStamped)();
  if (!franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_message_type_support_handle.typesupport_identifier) {
    franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &franka_msgs__msg__PoseStampedArray__rosidl_typesupport_introspection_c__PoseStampedArray_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
