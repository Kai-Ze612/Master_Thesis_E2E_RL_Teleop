// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mujoco_ros_msgs:msg/MocapState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mujoco_ros_msgs/msg/detail/mocap_state__rosidl_typesupport_introspection_c.h"
#include "mujoco_ros_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mujoco_ros_msgs/msg/detail/mocap_state__functions.h"
#include "mujoco_ros_msgs/msg/detail/mocap_state__struct.h"


// Include directives for member types
// Member `name`
#include "rosidl_runtime_c/string_functions.h"
// Member `pose`
#include "geometry_msgs/msg/pose_stamped.h"
// Member `pose`
#include "geometry_msgs/msg/detail/pose_stamped__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mujoco_ros_msgs__msg__MocapState__init(message_memory);
}

void mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_fini_function(void * message_memory)
{
  mujoco_ros_msgs__msg__MocapState__fini(message_memory);
}

size_t mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__size_function__MocapState__name(
  const void * untyped_member)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return member->size;
}

const void * mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_const_function__MocapState__name(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_function__MocapState__name(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__fetch_function__MocapState__name(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const rosidl_runtime_c__String * item =
    ((const rosidl_runtime_c__String *)
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_const_function__MocapState__name(untyped_member, index));
  rosidl_runtime_c__String * value =
    (rosidl_runtime_c__String *)(untyped_value);
  *value = *item;
}

void mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__assign_function__MocapState__name(
  void * untyped_member, size_t index, const void * untyped_value)
{
  rosidl_runtime_c__String * item =
    ((rosidl_runtime_c__String *)
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_function__MocapState__name(untyped_member, index));
  const rosidl_runtime_c__String * value =
    (const rosidl_runtime_c__String *)(untyped_value);
  *item = *value;
}

bool mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__resize_function__MocapState__name(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  rosidl_runtime_c__String__Sequence__fini(member);
  return rosidl_runtime_c__String__Sequence__init(member, size);
}

size_t mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__size_function__MocapState__pose(
  const void * untyped_member)
{
  const geometry_msgs__msg__PoseStamped__Sequence * member =
    (const geometry_msgs__msg__PoseStamped__Sequence *)(untyped_member);
  return member->size;
}

const void * mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_const_function__MocapState__pose(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__PoseStamped__Sequence * member =
    (const geometry_msgs__msg__PoseStamped__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_function__MocapState__pose(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__PoseStamped__Sequence * member =
    (geometry_msgs__msg__PoseStamped__Sequence *)(untyped_member);
  return &member->data[index];
}

void mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__fetch_function__MocapState__pose(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__PoseStamped * item =
    ((const geometry_msgs__msg__PoseStamped *)
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_const_function__MocapState__pose(untyped_member, index));
  geometry_msgs__msg__PoseStamped * value =
    (geometry_msgs__msg__PoseStamped *)(untyped_value);
  *value = *item;
}

void mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__assign_function__MocapState__pose(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__PoseStamped * item =
    ((geometry_msgs__msg__PoseStamped *)
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_function__MocapState__pose(untyped_member, index));
  const geometry_msgs__msg__PoseStamped * value =
    (const geometry_msgs__msg__PoseStamped *)(untyped_value);
  *item = *value;
}

bool mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__resize_function__MocapState__pose(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__PoseStamped__Sequence * member =
    (geometry_msgs__msg__PoseStamped__Sequence *)(untyped_member);
  geometry_msgs__msg__PoseStamped__Sequence__fini(member);
  return geometry_msgs__msg__PoseStamped__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_message_member_array[2] = {
  {
    "name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__MocapState, name),  // bytes offset in struct
    NULL,  // default value
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__size_function__MocapState__name,  // size() function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_const_function__MocapState__name,  // get_const(index) function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_function__MocapState__name,  // get(index) function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__fetch_function__MocapState__name,  // fetch(index, &value) function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__assign_function__MocapState__name,  // assign(index, value) function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__resize_function__MocapState__name  // resize(index) function pointer
  },
  {
    "pose",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__MocapState, pose),  // bytes offset in struct
    NULL,  // default value
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__size_function__MocapState__pose,  // size() function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_const_function__MocapState__pose,  // get_const(index) function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__get_function__MocapState__pose,  // get(index) function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__fetch_function__MocapState__pose,  // fetch(index, &value) function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__assign_function__MocapState__pose,  // assign(index, value) function pointer
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__resize_function__MocapState__pose  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_message_members = {
  "mujoco_ros_msgs__msg",  // message namespace
  "MocapState",  // message name
  2,  // number of fields
  sizeof(mujoco_ros_msgs__msg__MocapState),
  mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_message_member_array,  // message members
  mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_init_function,  // function to initialize message memory (memory has to be allocated)
  mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_message_type_support_handle = {
  0,
  &mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mujoco_ros_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, msg, MocapState)() {
  mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PoseStamped)();
  if (!mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_message_type_support_handle.typesupport_identifier) {
    mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mujoco_ros_msgs__msg__MocapState__rosidl_typesupport_introspection_c__MocapState_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
