// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mujoco_ros_msgs:msg/GeomType.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/geom_type__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
mujoco_ros_msgs__msg__GeomType__init(mujoco_ros_msgs__msg__GeomType * msg)
{
  if (!msg) {
    return false;
  }
  // value
  return true;
}

void
mujoco_ros_msgs__msg__GeomType__fini(mujoco_ros_msgs__msg__GeomType * msg)
{
  if (!msg) {
    return;
  }
  // value
}

bool
mujoco_ros_msgs__msg__GeomType__are_equal(const mujoco_ros_msgs__msg__GeomType * lhs, const mujoco_ros_msgs__msg__GeomType * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // value
  if (lhs->value != rhs->value) {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__msg__GeomType__copy(
  const mujoco_ros_msgs__msg__GeomType * input,
  mujoco_ros_msgs__msg__GeomType * output)
{
  if (!input || !output) {
    return false;
  }
  // value
  output->value = input->value;
  return true;
}

mujoco_ros_msgs__msg__GeomType *
mujoco_ros_msgs__msg__GeomType__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__GeomType * msg = (mujoco_ros_msgs__msg__GeomType *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__GeomType), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__msg__GeomType));
  bool success = mujoco_ros_msgs__msg__GeomType__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__msg__GeomType__destroy(mujoco_ros_msgs__msg__GeomType * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__msg__GeomType__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__msg__GeomType__Sequence__init(mujoco_ros_msgs__msg__GeomType__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__GeomType * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__msg__GeomType *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__msg__GeomType), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__msg__GeomType__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__msg__GeomType__fini(&data[i - 1]);
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
mujoco_ros_msgs__msg__GeomType__Sequence__fini(mujoco_ros_msgs__msg__GeomType__Sequence * array)
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
      mujoco_ros_msgs__msg__GeomType__fini(&array->data[i]);
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

mujoco_ros_msgs__msg__GeomType__Sequence *
mujoco_ros_msgs__msg__GeomType__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__GeomType__Sequence * array = (mujoco_ros_msgs__msg__GeomType__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__GeomType__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__msg__GeomType__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__msg__GeomType__Sequence__destroy(mujoco_ros_msgs__msg__GeomType__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__msg__GeomType__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__msg__GeomType__Sequence__are_equal(const mujoco_ros_msgs__msg__GeomType__Sequence * lhs, const mujoco_ros_msgs__msg__GeomType__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__msg__GeomType__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__msg__GeomType__Sequence__copy(
  const mujoco_ros_msgs__msg__GeomType__Sequence * input,
  mujoco_ros_msgs__msg__GeomType__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__msg__GeomType);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__msg__GeomType * data =
      (mujoco_ros_msgs__msg__GeomType *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__msg__GeomType__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__msg__GeomType__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__msg__GeomType__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
