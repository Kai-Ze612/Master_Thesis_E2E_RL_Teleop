// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from multi_mode_control_msgs:msg/JointGoal.idl
// generated code does not contain a copyright notice
#include "multi_mode_control_msgs/msg/detail/joint_goal__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
multi_mode_control_msgs__msg__JointGoal__init(multi_mode_control_msgs__msg__JointGoal * msg)
{
  if (!msg) {
    return false;
  }
  // q
  // qd
  return true;
}

void
multi_mode_control_msgs__msg__JointGoal__fini(multi_mode_control_msgs__msg__JointGoal * msg)
{
  if (!msg) {
    return;
  }
  // q
  // qd
}

bool
multi_mode_control_msgs__msg__JointGoal__are_equal(const multi_mode_control_msgs__msg__JointGoal * lhs, const multi_mode_control_msgs__msg__JointGoal * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // q
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->q[i] != rhs->q[i]) {
      return false;
    }
  }
  // qd
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->qd[i] != rhs->qd[i]) {
      return false;
    }
  }
  return true;
}

bool
multi_mode_control_msgs__msg__JointGoal__copy(
  const multi_mode_control_msgs__msg__JointGoal * input,
  multi_mode_control_msgs__msg__JointGoal * output)
{
  if (!input || !output) {
    return false;
  }
  // q
  for (size_t i = 0; i < 7; ++i) {
    output->q[i] = input->q[i];
  }
  // qd
  for (size_t i = 0; i < 7; ++i) {
    output->qd[i] = input->qd[i];
  }
  return true;
}

multi_mode_control_msgs__msg__JointGoal *
multi_mode_control_msgs__msg__JointGoal__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__JointGoal * msg = (multi_mode_control_msgs__msg__JointGoal *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__JointGoal), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(multi_mode_control_msgs__msg__JointGoal));
  bool success = multi_mode_control_msgs__msg__JointGoal__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
multi_mode_control_msgs__msg__JointGoal__destroy(multi_mode_control_msgs__msg__JointGoal * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    multi_mode_control_msgs__msg__JointGoal__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
multi_mode_control_msgs__msg__JointGoal__Sequence__init(multi_mode_control_msgs__msg__JointGoal__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__JointGoal * data = NULL;

  if (size) {
    data = (multi_mode_control_msgs__msg__JointGoal *)allocator.zero_allocate(size, sizeof(multi_mode_control_msgs__msg__JointGoal), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = multi_mode_control_msgs__msg__JointGoal__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        multi_mode_control_msgs__msg__JointGoal__fini(&data[i - 1]);
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
multi_mode_control_msgs__msg__JointGoal__Sequence__fini(multi_mode_control_msgs__msg__JointGoal__Sequence * array)
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
      multi_mode_control_msgs__msg__JointGoal__fini(&array->data[i]);
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

multi_mode_control_msgs__msg__JointGoal__Sequence *
multi_mode_control_msgs__msg__JointGoal__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__JointGoal__Sequence * array = (multi_mode_control_msgs__msg__JointGoal__Sequence *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__JointGoal__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = multi_mode_control_msgs__msg__JointGoal__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
multi_mode_control_msgs__msg__JointGoal__Sequence__destroy(multi_mode_control_msgs__msg__JointGoal__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    multi_mode_control_msgs__msg__JointGoal__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
multi_mode_control_msgs__msg__JointGoal__Sequence__are_equal(const multi_mode_control_msgs__msg__JointGoal__Sequence * lhs, const multi_mode_control_msgs__msg__JointGoal__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!multi_mode_control_msgs__msg__JointGoal__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
multi_mode_control_msgs__msg__JointGoal__Sequence__copy(
  const multi_mode_control_msgs__msg__JointGoal__Sequence * input,
  multi_mode_control_msgs__msg__JointGoal__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(multi_mode_control_msgs__msg__JointGoal);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    multi_mode_control_msgs__msg__JointGoal * data =
      (multi_mode_control_msgs__msg__JointGoal *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!multi_mode_control_msgs__msg__JointGoal__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          multi_mode_control_msgs__msg__JointGoal__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!multi_mode_control_msgs__msg__JointGoal__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
