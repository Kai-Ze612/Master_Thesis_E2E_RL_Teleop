// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from panda_motion_generator_msgs:msg/SetTimeScaling.idl
// generated code does not contain a copyright notice
#include "panda_motion_generator_msgs/msg/detail/set_time_scaling__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
panda_motion_generator_msgs__msg__SetTimeScaling__init(panda_motion_generator_msgs__msg__SetTimeScaling * msg)
{
  if (!msg) {
    return false;
  }
  // steps
  // scaling
  return true;
}

void
panda_motion_generator_msgs__msg__SetTimeScaling__fini(panda_motion_generator_msgs__msg__SetTimeScaling * msg)
{
  if (!msg) {
    return;
  }
  // steps
  // scaling
}

bool
panda_motion_generator_msgs__msg__SetTimeScaling__are_equal(const panda_motion_generator_msgs__msg__SetTimeScaling * lhs, const panda_motion_generator_msgs__msg__SetTimeScaling * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // steps
  if (lhs->steps != rhs->steps) {
    return false;
  }
  // scaling
  if (lhs->scaling != rhs->scaling) {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__msg__SetTimeScaling__copy(
  const panda_motion_generator_msgs__msg__SetTimeScaling * input,
  panda_motion_generator_msgs__msg__SetTimeScaling * output)
{
  if (!input || !output) {
    return false;
  }
  // steps
  output->steps = input->steps;
  // scaling
  output->scaling = input->scaling;
  return true;
}

panda_motion_generator_msgs__msg__SetTimeScaling *
panda_motion_generator_msgs__msg__SetTimeScaling__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__msg__SetTimeScaling * msg = (panda_motion_generator_msgs__msg__SetTimeScaling *)allocator.allocate(sizeof(panda_motion_generator_msgs__msg__SetTimeScaling), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__msg__SetTimeScaling));
  bool success = panda_motion_generator_msgs__msg__SetTimeScaling__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__msg__SetTimeScaling__destroy(panda_motion_generator_msgs__msg__SetTimeScaling * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__msg__SetTimeScaling__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__init(panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__msg__SetTimeScaling * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__msg__SetTimeScaling *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__msg__SetTimeScaling), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__msg__SetTimeScaling__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__msg__SetTimeScaling__fini(&data[i - 1]);
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
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__fini(panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * array)
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
      panda_motion_generator_msgs__msg__SetTimeScaling__fini(&array->data[i]);
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

panda_motion_generator_msgs__msg__SetTimeScaling__Sequence *
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * array = (panda_motion_generator_msgs__msg__SetTimeScaling__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__msg__SetTimeScaling__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__destroy(panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__are_equal(const panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * lhs, const panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__msg__SetTimeScaling__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__msg__SetTimeScaling__Sequence__copy(
  const panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * input,
  panda_motion_generator_msgs__msg__SetTimeScaling__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__msg__SetTimeScaling);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__msg__SetTimeScaling * data =
      (panda_motion_generator_msgs__msg__SetTimeScaling *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__msg__SetTimeScaling__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__msg__SetTimeScaling__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__msg__SetTimeScaling__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
