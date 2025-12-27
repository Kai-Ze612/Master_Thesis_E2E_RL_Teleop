// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from multi_mode_control_msgs:msg/CartesianImpedanceGoal.idl
// generated code does not contain a copyright notice
#include "multi_mode_control_msgs/msg/detail/cartesian_impedance_goal__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `pose`
#include "geometry_msgs/msg/detail/pose__functions.h"

bool
multi_mode_control_msgs__msg__CartesianImpedanceGoal__init(multi_mode_control_msgs__msg__CartesianImpedanceGoal * msg)
{
  if (!msg) {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__Pose__init(&msg->pose)) {
    multi_mode_control_msgs__msg__CartesianImpedanceGoal__fini(msg);
    return false;
  }
  // q_n
  return true;
}

void
multi_mode_control_msgs__msg__CartesianImpedanceGoal__fini(multi_mode_control_msgs__msg__CartesianImpedanceGoal * msg)
{
  if (!msg) {
    return;
  }
  // pose
  geometry_msgs__msg__Pose__fini(&msg->pose);
  // q_n
}

bool
multi_mode_control_msgs__msg__CartesianImpedanceGoal__are_equal(const multi_mode_control_msgs__msg__CartesianImpedanceGoal * lhs, const multi_mode_control_msgs__msg__CartesianImpedanceGoal * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->pose), &(rhs->pose)))
  {
    return false;
  }
  // q_n
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->q_n[i] != rhs->q_n[i]) {
      return false;
    }
  }
  return true;
}

bool
multi_mode_control_msgs__msg__CartesianImpedanceGoal__copy(
  const multi_mode_control_msgs__msg__CartesianImpedanceGoal * input,
  multi_mode_control_msgs__msg__CartesianImpedanceGoal * output)
{
  if (!input || !output) {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->pose), &(output->pose)))
  {
    return false;
  }
  // q_n
  for (size_t i = 0; i < 7; ++i) {
    output->q_n[i] = input->q_n[i];
  }
  return true;
}

multi_mode_control_msgs__msg__CartesianImpedanceGoal *
multi_mode_control_msgs__msg__CartesianImpedanceGoal__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__CartesianImpedanceGoal * msg = (multi_mode_control_msgs__msg__CartesianImpedanceGoal *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__CartesianImpedanceGoal), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(multi_mode_control_msgs__msg__CartesianImpedanceGoal));
  bool success = multi_mode_control_msgs__msg__CartesianImpedanceGoal__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
multi_mode_control_msgs__msg__CartesianImpedanceGoal__destroy(multi_mode_control_msgs__msg__CartesianImpedanceGoal * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    multi_mode_control_msgs__msg__CartesianImpedanceGoal__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence__init(multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__CartesianImpedanceGoal * data = NULL;

  if (size) {
    data = (multi_mode_control_msgs__msg__CartesianImpedanceGoal *)allocator.zero_allocate(size, sizeof(multi_mode_control_msgs__msg__CartesianImpedanceGoal), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = multi_mode_control_msgs__msg__CartesianImpedanceGoal__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        multi_mode_control_msgs__msg__CartesianImpedanceGoal__fini(&data[i - 1]);
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
multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence__fini(multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence * array)
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
      multi_mode_control_msgs__msg__CartesianImpedanceGoal__fini(&array->data[i]);
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

multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence *
multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence * array = (multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence__destroy(multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence__are_equal(const multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence * lhs, const multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!multi_mode_control_msgs__msg__CartesianImpedanceGoal__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence__copy(
  const multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence * input,
  multi_mode_control_msgs__msg__CartesianImpedanceGoal__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(multi_mode_control_msgs__msg__CartesianImpedanceGoal);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    multi_mode_control_msgs__msg__CartesianImpedanceGoal * data =
      (multi_mode_control_msgs__msg__CartesianImpedanceGoal *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!multi_mode_control_msgs__msg__CartesianImpedanceGoal__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          multi_mode_control_msgs__msg__CartesianImpedanceGoal__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!multi_mode_control_msgs__msg__CartesianImpedanceGoal__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
