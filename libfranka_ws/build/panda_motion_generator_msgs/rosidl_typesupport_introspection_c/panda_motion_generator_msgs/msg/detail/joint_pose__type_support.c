// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from panda_motion_generator_msgs:msg/JointPose.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "panda_motion_generator_msgs/msg/detail/joint_pose__rosidl_typesupport_introspection_c.h"
#include "panda_motion_generator_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "panda_motion_generator_msgs/msg/detail/joint_pose__functions.h"
#include "panda_motion_generator_msgs/msg/detail/joint_pose__struct.h"


// Include directives for member types
// Member `q`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  panda_motion_generator_msgs__msg__JointPose__init(message_memory);
}

void panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_fini_function(void * message_memory)
{
  panda_motion_generator_msgs__msg__JointPose__fini(message_memory);
}

size_t panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__size_function__JointPose__q(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__get_const_function__JointPose__q(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__get_function__JointPose__q(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__fetch_function__JointPose__q(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__get_const_function__JointPose__q(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__assign_function__JointPose__q(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__get_function__JointPose__q(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__resize_function__JointPose__q(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_message_member_array[1] = {
  {
    "q",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(panda_motion_generator_msgs__msg__JointPose, q),  // bytes offset in struct
    NULL,  // default value
    panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__size_function__JointPose__q,  // size() function pointer
    panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__get_const_function__JointPose__q,  // get_const(index) function pointer
    panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__get_function__JointPose__q,  // get(index) function pointer
    panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__fetch_function__JointPose__q,  // fetch(index, &value) function pointer
    panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__assign_function__JointPose__q,  // assign(index, value) function pointer
    panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__resize_function__JointPose__q  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_message_members = {
  "panda_motion_generator_msgs__msg",  // message namespace
  "JointPose",  // message name
  1,  // number of fields
  sizeof(panda_motion_generator_msgs__msg__JointPose),
  panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_message_member_array,  // message members
  panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_init_function,  // function to initialize message memory (memory has to be allocated)
  panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_message_type_support_handle = {
  0,
  &panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_panda_motion_generator_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, panda_motion_generator_msgs, msg, JointPose)() {
  if (!panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_message_type_support_handle.typesupport_identifier) {
    panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &panda_motion_generator_msgs__msg__JointPose__rosidl_typesupport_introspection_c__JointPose_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
