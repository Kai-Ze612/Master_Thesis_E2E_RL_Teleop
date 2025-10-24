// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mujoco_ros_msgs:msg/SensorNoiseModel.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mujoco_ros_msgs/msg/detail/sensor_noise_model__rosidl_typesupport_introspection_c.h"
#include "mujoco_ros_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mujoco_ros_msgs/msg/detail/sensor_noise_model__functions.h"
#include "mujoco_ros_msgs/msg/detail/sensor_noise_model__struct.h"


// Include directives for member types
// Member `sensor_name`
#include "rosidl_runtime_c/string_functions.h"
// Member `mean`
// Member `std`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mujoco_ros_msgs__msg__SensorNoiseModel__init(message_memory);
}

void mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_fini_function(void * message_memory)
{
  mujoco_ros_msgs__msg__SensorNoiseModel__fini(message_memory);
}

size_t mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__size_function__SensorNoiseModel__mean(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_const_function__SensorNoiseModel__mean(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_function__SensorNoiseModel__mean(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__fetch_function__SensorNoiseModel__mean(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_const_function__SensorNoiseModel__mean(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__assign_function__SensorNoiseModel__mean(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_function__SensorNoiseModel__mean(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__resize_function__SensorNoiseModel__mean(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__size_function__SensorNoiseModel__std(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_const_function__SensorNoiseModel__std(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_function__SensorNoiseModel__std(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__fetch_function__SensorNoiseModel__std(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_const_function__SensorNoiseModel__std(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__assign_function__SensorNoiseModel__std(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_function__SensorNoiseModel__std(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__resize_function__SensorNoiseModel__std(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_message_member_array[4] = {
  {
    "sensor_name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__SensorNoiseModel, sensor_name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mean",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__SensorNoiseModel, mean),  // bytes offset in struct
    NULL,  // default value
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__size_function__SensorNoiseModel__mean,  // size() function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_const_function__SensorNoiseModel__mean,  // get_const(index) function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_function__SensorNoiseModel__mean,  // get(index) function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__fetch_function__SensorNoiseModel__mean,  // fetch(index, &value) function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__assign_function__SensorNoiseModel__mean,  // assign(index, value) function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__resize_function__SensorNoiseModel__mean  // resize(index) function pointer
  },
  {
    "std",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__SensorNoiseModel, std),  // bytes offset in struct
    NULL,  // default value
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__size_function__SensorNoiseModel__std,  // size() function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_const_function__SensorNoiseModel__std,  // get_const(index) function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__get_function__SensorNoiseModel__std,  // get(index) function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__fetch_function__SensorNoiseModel__std,  // fetch(index, &value) function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__assign_function__SensorNoiseModel__std,  // assign(index, value) function pointer
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__resize_function__SensorNoiseModel__std  // resize(index) function pointer
  },
  {
    "set_flag",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mujoco_ros_msgs__msg__SensorNoiseModel, set_flag),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_message_members = {
  "mujoco_ros_msgs__msg",  // message namespace
  "SensorNoiseModel",  // message name
  4,  // number of fields
  sizeof(mujoco_ros_msgs__msg__SensorNoiseModel),
  mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_message_member_array,  // message members
  mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_init_function,  // function to initialize message memory (memory has to be allocated)
  mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_message_type_support_handle = {
  0,
  &mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mujoco_ros_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, msg, SensorNoiseModel)() {
  if (!mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_message_type_support_handle.typesupport_identifier) {
    mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mujoco_ros_msgs__msg__SensorNoiseModel__rosidl_typesupport_introspection_c__SensorNoiseModel_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
