// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from franka_msgs:msg/PoseStampedArray.idl
// generated code does not contain a copyright notice
#include "franka_msgs/msg/detail/pose_stamped_array__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `stampedposes`
#include "geometry_msgs/msg/detail/pose_stamped__functions.h"

bool
franka_msgs__msg__PoseStampedArray__init(franka_msgs__msg__PoseStampedArray * msg)
{
  if (!msg) {
    return false;
  }
  // stampedposes
  if (!geometry_msgs__msg__PoseStamped__Sequence__init(&msg->stampedposes, 0)) {
    franka_msgs__msg__PoseStampedArray__fini(msg);
    return false;
  }
  return true;
}

void
franka_msgs__msg__PoseStampedArray__fini(franka_msgs__msg__PoseStampedArray * msg)
{
  if (!msg) {
    return;
  }
  // stampedposes
  geometry_msgs__msg__PoseStamped__Sequence__fini(&msg->stampedposes);
}

bool
franka_msgs__msg__PoseStampedArray__are_equal(const franka_msgs__msg__PoseStampedArray * lhs, const franka_msgs__msg__PoseStampedArray * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // stampedposes
  if (!geometry_msgs__msg__PoseStamped__Sequence__are_equal(
      &(lhs->stampedposes), &(rhs->stampedposes)))
  {
    return false;
  }
  return true;
}

bool
franka_msgs__msg__PoseStampedArray__copy(
  const franka_msgs__msg__PoseStampedArray * input,
  franka_msgs__msg__PoseStampedArray * output)
{
  if (!input || !output) {
    return false;
  }
  // stampedposes
  if (!geometry_msgs__msg__PoseStamped__Sequence__copy(
      &(input->stampedposes), &(output->stampedposes)))
  {
    return false;
  }
  return true;
}

franka_msgs__msg__PoseStampedArray *
franka_msgs__msg__PoseStampedArray__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  franka_msgs__msg__PoseStampedArray * msg = (franka_msgs__msg__PoseStampedArray *)allocator.allocate(sizeof(franka_msgs__msg__PoseStampedArray), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(franka_msgs__msg__PoseStampedArray));
  bool success = franka_msgs__msg__PoseStampedArray__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
franka_msgs__msg__PoseStampedArray__destroy(franka_msgs__msg__PoseStampedArray * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    franka_msgs__msg__PoseStampedArray__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
franka_msgs__msg__PoseStampedArray__Sequence__init(franka_msgs__msg__PoseStampedArray__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  franka_msgs__msg__PoseStampedArray * data = NULL;

  if (size) {
    data = (franka_msgs__msg__PoseStampedArray *)allocator.zero_allocate(size, sizeof(franka_msgs__msg__PoseStampedArray), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = franka_msgs__msg__PoseStampedArray__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        franka_msgs__msg__PoseStampedArray__fini(&data[i - 1]);
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
franka_msgs__msg__PoseStampedArray__Sequence__fini(franka_msgs__msg__PoseStampedArray__Sequence * array)
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
      franka_msgs__msg__PoseStampedArray__fini(&array->data[i]);
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

franka_msgs__msg__PoseStampedArray__Sequence *
franka_msgs__msg__PoseStampedArray__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  franka_msgs__msg__PoseStampedArray__Sequence * array = (franka_msgs__msg__PoseStampedArray__Sequence *)allocator.allocate(sizeof(franka_msgs__msg__PoseStampedArray__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = franka_msgs__msg__PoseStampedArray__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
franka_msgs__msg__PoseStampedArray__Sequence__destroy(franka_msgs__msg__PoseStampedArray__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    franka_msgs__msg__PoseStampedArray__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
franka_msgs__msg__PoseStampedArray__Sequence__are_equal(const franka_msgs__msg__PoseStampedArray__Sequence * lhs, const franka_msgs__msg__PoseStampedArray__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!franka_msgs__msg__PoseStampedArray__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
franka_msgs__msg__PoseStampedArray__Sequence__copy(
  const franka_msgs__msg__PoseStampedArray__Sequence * input,
  franka_msgs__msg__PoseStampedArray__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(franka_msgs__msg__PoseStampedArray);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    franka_msgs__msg__PoseStampedArray * data =
      (franka_msgs__msg__PoseStampedArray *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!franka_msgs__msg__PoseStampedArray__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          franka_msgs__msg__PoseStampedArray__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!franka_msgs__msg__PoseStampedArray__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
