// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from panda_motion_generator_msgs:msg/SimpleActionResult.idl
// generated code does not contain a copyright notice
#include "panda_motion_generator_msgs/msg/detail/simple_action_result__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `error`
#include "rosidl_runtime_c/string_functions.h"

bool
panda_motion_generator_msgs__msg__SimpleActionResult__init(panda_motion_generator_msgs__msg__SimpleActionResult * msg)
{
  if (!msg) {
    return false;
  }
  // state
  // error
  if (!rosidl_runtime_c__String__init(&msg->error)) {
    panda_motion_generator_msgs__msg__SimpleActionResult__fini(msg);
    return false;
  }
  return true;
}

void
panda_motion_generator_msgs__msg__SimpleActionResult__fini(panda_motion_generator_msgs__msg__SimpleActionResult * msg)
{
  if (!msg) {
    return;
  }
  // state
  // error
  rosidl_runtime_c__String__fini(&msg->error);
}

bool
panda_motion_generator_msgs__msg__SimpleActionResult__are_equal(const panda_motion_generator_msgs__msg__SimpleActionResult * lhs, const panda_motion_generator_msgs__msg__SimpleActionResult * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // state
  if (lhs->state != rhs->state) {
    return false;
  }
  // error
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->error), &(rhs->error)))
  {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__msg__SimpleActionResult__copy(
  const panda_motion_generator_msgs__msg__SimpleActionResult * input,
  panda_motion_generator_msgs__msg__SimpleActionResult * output)
{
  if (!input || !output) {
    return false;
  }
  // state
  output->state = input->state;
  // error
  if (!rosidl_runtime_c__String__copy(
      &(input->error), &(output->error)))
  {
    return false;
  }
  return true;
}

panda_motion_generator_msgs__msg__SimpleActionResult *
panda_motion_generator_msgs__msg__SimpleActionResult__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__msg__SimpleActionResult * msg = (panda_motion_generator_msgs__msg__SimpleActionResult *)allocator.allocate(sizeof(panda_motion_generator_msgs__msg__SimpleActionResult), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__msg__SimpleActionResult));
  bool success = panda_motion_generator_msgs__msg__SimpleActionResult__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__msg__SimpleActionResult__destroy(panda_motion_generator_msgs__msg__SimpleActionResult * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__msg__SimpleActionResult__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__msg__SimpleActionResult__Sequence__init(panda_motion_generator_msgs__msg__SimpleActionResult__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__msg__SimpleActionResult * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__msg__SimpleActionResult *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__msg__SimpleActionResult), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__msg__SimpleActionResult__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__msg__SimpleActionResult__fini(&data[i - 1]);
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
panda_motion_generator_msgs__msg__SimpleActionResult__Sequence__fini(panda_motion_generator_msgs__msg__SimpleActionResult__Sequence * array)
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
      panda_motion_generator_msgs__msg__SimpleActionResult__fini(&array->data[i]);
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

panda_motion_generator_msgs__msg__SimpleActionResult__Sequence *
panda_motion_generator_msgs__msg__SimpleActionResult__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__msg__SimpleActionResult__Sequence * array = (panda_motion_generator_msgs__msg__SimpleActionResult__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__msg__SimpleActionResult__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__msg__SimpleActionResult__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__msg__SimpleActionResult__Sequence__destroy(panda_motion_generator_msgs__msg__SimpleActionResult__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__msg__SimpleActionResult__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__msg__SimpleActionResult__Sequence__are_equal(const panda_motion_generator_msgs__msg__SimpleActionResult__Sequence * lhs, const panda_motion_generator_msgs__msg__SimpleActionResult__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__msg__SimpleActionResult__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__msg__SimpleActionResult__Sequence__copy(
  const panda_motion_generator_msgs__msg__SimpleActionResult__Sequence * input,
  panda_motion_generator_msgs__msg__SimpleActionResult__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__msg__SimpleActionResult);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__msg__SimpleActionResult * data =
      (panda_motion_generator_msgs__msg__SimpleActionResult *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__msg__SimpleActionResult__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__msg__SimpleActionResult__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__msg__SimpleActionResult__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
