// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mujoco_ros_msgs:msg/EqualityConstraintParameters.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mujoco_ros_msgs/msg/detail/equality_constraint_parameters__rosidl_typesupport_introspection_c.h"
#include "mujoco_ros_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mujoco_ros_msgs/msg/detail/equality_constraint_parameters__functions.h"
#include "mujoco_ros_msgs/msg/detail/equality_constraint_parameters__struct.h"


// Include directives for member types
// Member `name`
// Member `class_param`
// Member `element1`
// Member `element2`
#include "rosidl_runtime_c/string_functions.h"
// Member `type`
#include "mujoco_ros_msgs/msg/equality_constraint_type.h"
// Member `type`
#include "mujoco_ros_msgs/msg/detail/equality_constraint_type__rosidl_typesupport_introspection_c.h"
// Member `solver_parameters`
#include "mujoco_ros_msgs/msg/solver_parameters.h"
// Member `solver_parameters`
#include "mujoco_ros_msgs/msg/detail/solver_parameters__rosidl_typesupport_introspection_c.h"
// Member `anchor`
#include "geometry_msgs/msg/vector3.h"
// Member `anchor`
#include "geometry_msgs/msg/detail/vector3__rosidl_typesupport_introspection_c.h"
// Member `relpose`
#include "geometry_msgs/msg/pose.h"
// Member `relpose`
#include "geometry_msgs/msg/detail/pose__rosidl_typesupport_introspection_c.h"
// Member `polycoef`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mujoco_ros_msgs__msg__EqualityConstraintParameters__init(message_memory);
}

void mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_fini_function(void * message_memory)
{
  mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(message_memory);
}

size_t mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__size_function__EqualityConstraintParameters__polycoef(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__get_const_function__EqualityConstraintParameters__polycoef(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__get_function__EqualityConstraintParameters__polycoef(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__fetch_function__EqualityConstraintParameters__polycoef(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__get_const_function__EqualityConstraintParameters__polycoef(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__assign_function__EqualityConstraintParameters__polycoef(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__get_function__EqualityConstraintParameters__polycoef(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__resize_function__EqualityConstraintParameters__polycoef(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_member_array[11] = {
  {
    "name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, type),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "solver_parameters",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, solver_parameters),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "active",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, active),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "class_param",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, class_param),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "element1",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, element1),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "element2",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, element2),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "torquescale",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, torquescale),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "anchor",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, anchor),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "relpose",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, relpose),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "polycoef",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__EqualityConstraintParameters, polycoef),  // bytes offset in struct
    NULL,  // default value
    mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__size_function__EqualityConstraintParameters__polycoef,  // size() function pointer
    mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__get_const_function__EqualityConstraintParameters__polycoef,  // get_const(index) function pointer
    mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__get_function__EqualityConstraintParameters__polycoef,  // get(index) function pointer
    mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__fetch_function__EqualityConstraintParameters__polycoef,  // fetch(index, &value) function pointer
    mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__assign_function__EqualityConstraintParameters__polycoef,  // assign(index, value) function pointer
    mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__resize_function__EqualityConstraintParameters__polycoef  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_members = {
  "mujoco_ros_msgs__msg",  // message namespace
  "EqualityConstraintParameters",  // message name
  11,  // number of fields
  sizeof(mujoco_ros_msgs__msg__EqualityConstraintParameters),
  mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_member_array,  // message members
  mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_init_function,  // function to initialize message memory (memory has to be allocated)
  mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_type_support_handle = {
  0,
  &mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mujoco_ros_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, msg, EqualityConstraintParameters)() {
  mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, msg, EqualityConstraintType)();
  mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, msg, SolverParameters)();
  mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_member_array[8].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_member_array[9].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  if (!mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_type_support_handle.typesupport_identifier) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mujoco_ros_msgs__msg__EqualityConstraintParameters__rosidl_typesupport_introspection_c__EqualityConstraintParameters_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
