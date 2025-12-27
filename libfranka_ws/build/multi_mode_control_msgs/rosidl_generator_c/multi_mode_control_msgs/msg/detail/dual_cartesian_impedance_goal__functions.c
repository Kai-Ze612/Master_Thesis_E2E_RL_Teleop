// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from multi_mode_control_msgs:msg/DualCartesianImpedanceGoal.idl
// generated code does not contain a copyright notice
#include "multi_mode_control_msgs/msg/detail/dual_cartesian_impedance_goal__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `l_pose`
// Member `r_pose`
#include "geometry_msgs/msg/detail/pose__functions.h"

bool
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__init(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * msg)
{
  if (!msg) {
    return false;
  }
  // l_pose
  if (!geometry_msgs__msg__Pose__init(&msg->l_pose)) {
    multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__fini(msg);
    return false;
  }
  // r_pose
  if (!geometry_msgs__msg__Pose__init(&msg->r_pose)) {
    multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__fini(msg);
    return false;
  }
  // l_q_n
  // r_q_n
  return true;
}

void
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__fini(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * msg)
{
  if (!msg) {
    return;
  }
  // l_pose
  geometry_msgs__msg__Pose__fini(&msg->l_pose);
  // r_pose
  geometry_msgs__msg__Pose__fini(&msg->r_pose);
  // l_q_n
  // r_q_n
}

bool
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__are_equal(const multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * lhs, const multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // l_pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->l_pose), &(rhs->l_pose)))
  {
    return false;
  }
  // r_pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->r_pose), &(rhs->r_pose)))
  {
    return false;
  }
  // l_q_n
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->l_q_n[i] != rhs->l_q_n[i]) {
      return false;
    }
  }
  // r_q_n
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->r_q_n[i] != rhs->r_q_n[i]) {
      return false;
    }
  }
  return true;
}

bool
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__copy(
  const multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * input,
  multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * output)
{
  if (!input || !output) {
    return false;
  }
  // l_pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->l_pose), &(output->l_pose)))
  {
    return false;
  }
  // r_pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->r_pose), &(output->r_pose)))
  {
    return false;
  }
  // l_q_n
  for (size_t i = 0; i < 7; ++i) {
    output->l_q_n[i] = input->l_q_n[i];
  }
  // r_q_n
  for (size_t i = 0; i < 7; ++i) {
    output->r_q_n[i] = input->r_q_n[i];
  }
  return true;
}

multi_mode_control_msgs__msg__DualCartesianImpedanceGoal *
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * msg = (multi_mode_control_msgs__msg__DualCartesianImpedanceGoal *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal));
  bool success = multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__destroy(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence__init(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * data = NULL;

  if (size) {
    data = (multi_mode_control_msgs__msg__DualCartesianImpedanceGoal *)allocator.zero_allocate(size, sizeof(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__fini(&data[i - 1]);
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
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence__fini(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence * array)
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
      multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__fini(&array->data[i]);
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

multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence *
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence * array = (multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence__destroy(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence__are_equal(const multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence * lhs, const multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence__copy(
  const multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence * input,
  multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(multi_mode_control_msgs__msg__DualCartesianImpedanceGoal);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    multi_mode_control_msgs__msg__DualCartesianImpedanceGoal * data =
      (multi_mode_control_msgs__msg__DualCartesianImpedanceGoal *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!multi_mode_control_msgs__msg__DualCartesianImpedanceGoal__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
