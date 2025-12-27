// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mujoco_ros_msgs:msg/EqualityConstraintParameters.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/equality_constraint_parameters__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `name`
// Member `class_param`
// Member `element1`
// Member `element2`
#include "rosidl_runtime_c/string_functions.h"
// Member `type`
#include "mujoco_ros_msgs/msg/detail/equality_constraint_type__functions.h"
// Member `solver_parameters`
#include "mujoco_ros_msgs/msg/detail/solver_parameters__functions.h"
// Member `anchor`
#include "geometry_msgs/msg/detail/vector3__functions.h"
// Member `relpose`
#include "geometry_msgs/msg/detail/pose__functions.h"
// Member `polycoef`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
mujoco_ros_msgs__msg__EqualityConstraintParameters__init(mujoco_ros_msgs__msg__EqualityConstraintParameters * msg)
{
  if (!msg) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__init(&msg->name)) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
    return false;
  }
  // type
  if (!mujoco_ros_msgs__msg__EqualityConstraintType__init(&msg->type)) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
    return false;
  }
  // solver_parameters
  if (!mujoco_ros_msgs__msg__SolverParameters__init(&msg->solver_parameters)) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
    return false;
  }
  // active
  // class_param
  if (!rosidl_runtime_c__String__init(&msg->class_param)) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
    return false;
  }
  // element1
  if (!rosidl_runtime_c__String__init(&msg->element1)) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
    return false;
  }
  // element2
  if (!rosidl_runtime_c__String__init(&msg->element2)) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
    return false;
  }
  // torquescale
  // anchor
  if (!geometry_msgs__msg__Vector3__init(&msg->anchor)) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
    return false;
  }
  // relpose
  if (!geometry_msgs__msg__Pose__init(&msg->relpose)) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
    return false;
  }
  // polycoef
  if (!rosidl_runtime_c__double__Sequence__init(&msg->polycoef, 0)) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
    return false;
  }
  return true;
}

void
mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(mujoco_ros_msgs__msg__EqualityConstraintParameters * msg)
{
  if (!msg) {
    return;
  }
  // name
  rosidl_runtime_c__String__fini(&msg->name);
  // type
  mujoco_ros_msgs__msg__EqualityConstraintType__fini(&msg->type);
  // solver_parameters
  mujoco_ros_msgs__msg__SolverParameters__fini(&msg->solver_parameters);
  // active
  // class_param
  rosidl_runtime_c__String__fini(&msg->class_param);
  // element1
  rosidl_runtime_c__String__fini(&msg->element1);
  // element2
  rosidl_runtime_c__String__fini(&msg->element2);
  // torquescale
  // anchor
  geometry_msgs__msg__Vector3__fini(&msg->anchor);
  // relpose
  geometry_msgs__msg__Pose__fini(&msg->relpose);
  // polycoef
  rosidl_runtime_c__double__Sequence__fini(&msg->polycoef);
}

bool
mujoco_ros_msgs__msg__EqualityConstraintParameters__are_equal(const mujoco_ros_msgs__msg__EqualityConstraintParameters * lhs, const mujoco_ros_msgs__msg__EqualityConstraintParameters * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->name), &(rhs->name)))
  {
    return false;
  }
  // type
  if (!mujoco_ros_msgs__msg__EqualityConstraintType__are_equal(
      &(lhs->type), &(rhs->type)))
  {
    return false;
  }
  // solver_parameters
  if (!mujoco_ros_msgs__msg__SolverParameters__are_equal(
      &(lhs->solver_parameters), &(rhs->solver_parameters)))
  {
    return false;
  }
  // active
  if (lhs->active != rhs->active) {
    return false;
  }
  // class_param
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->class_param), &(rhs->class_param)))
  {
    return false;
  }
  // element1
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->element1), &(rhs->element1)))
  {
    return false;
  }
  // element2
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->element2), &(rhs->element2)))
  {
    return false;
  }
  // torquescale
  if (lhs->torquescale != rhs->torquescale) {
    return false;
  }
  // anchor
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->anchor), &(rhs->anchor)))
  {
    return false;
  }
  // relpose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->relpose), &(rhs->relpose)))
  {
    return false;
  }
  // polycoef
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->polycoef), &(rhs->polycoef)))
  {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__msg__EqualityConstraintParameters__copy(
  const mujoco_ros_msgs__msg__EqualityConstraintParameters * input,
  mujoco_ros_msgs__msg__EqualityConstraintParameters * output)
{
  if (!input || !output) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__copy(
      &(input->name), &(output->name)))
  {
    return false;
  }
  // type
  if (!mujoco_ros_msgs__msg__EqualityConstraintType__copy(
      &(input->type), &(output->type)))
  {
    return false;
  }
  // solver_parameters
  if (!mujoco_ros_msgs__msg__SolverParameters__copy(
      &(input->solver_parameters), &(output->solver_parameters)))
  {
    return false;
  }
  // active
  output->active = input->active;
  // class_param
  if (!rosidl_runtime_c__String__copy(
      &(input->class_param), &(output->class_param)))
  {
    return false;
  }
  // element1
  if (!rosidl_runtime_c__String__copy(
      &(input->element1), &(output->element1)))
  {
    return false;
  }
  // element2
  if (!rosidl_runtime_c__String__copy(
      &(input->element2), &(output->element2)))
  {
    return false;
  }
  // torquescale
  output->torquescale = input->torquescale;
  // anchor
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->anchor), &(output->anchor)))
  {
    return false;
  }
  // relpose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->relpose), &(output->relpose)))
  {
    return false;
  }
  // polycoef
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->polycoef), &(output->polycoef)))
  {
    return false;
  }
  return true;
}

mujoco_ros_msgs__msg__EqualityConstraintParameters *
mujoco_ros_msgs__msg__EqualityConstraintParameters__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__EqualityConstraintParameters * msg = (mujoco_ros_msgs__msg__EqualityConstraintParameters *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__EqualityConstraintParameters), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__msg__EqualityConstraintParameters));
  bool success = mujoco_ros_msgs__msg__EqualityConstraintParameters__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__msg__EqualityConstraintParameters__destroy(mujoco_ros_msgs__msg__EqualityConstraintParameters * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__init(mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__EqualityConstraintParameters * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__msg__EqualityConstraintParameters *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__msg__EqualityConstraintParameters), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__msg__EqualityConstraintParameters__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__fini(mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence *
mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * array = (mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__destroy(mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__are_equal(const mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * lhs, const mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__msg__EqualityConstraintParameters__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__copy(
  const mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * input,
  mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__msg__EqualityConstraintParameters);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__msg__EqualityConstraintParameters * data =
      (mujoco_ros_msgs__msg__EqualityConstraintParameters *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__msg__EqualityConstraintParameters__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__msg__EqualityConstraintParameters__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__msg__EqualityConstraintParameters__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
