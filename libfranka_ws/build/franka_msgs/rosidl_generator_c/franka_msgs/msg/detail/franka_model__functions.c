// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from franka_msgs:msg/FrankaModel.idl
// generated code does not contain a copyright notice
#include "franka_msgs/msg/detail/franka_model__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
franka_msgs__msg__FrankaModel__init(franka_msgs__msg__FrankaModel * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    franka_msgs__msg__FrankaModel__fini(msg);
    return false;
  }
  // coriolis
  // ee_zero_jacobian
  // ee_body_jacobian
  // mass
  return true;
}

void
franka_msgs__msg__FrankaModel__fini(franka_msgs__msg__FrankaModel * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // coriolis
  // ee_zero_jacobian
  // ee_body_jacobian
  // mass
}

bool
franka_msgs__msg__FrankaModel__are_equal(const franka_msgs__msg__FrankaModel * lhs, const franka_msgs__msg__FrankaModel * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // coriolis
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->coriolis[i] != rhs->coriolis[i]) {
      return false;
    }
  }
  // ee_zero_jacobian
  for (size_t i = 0; i < 42; ++i) {
    if (lhs->ee_zero_jacobian[i] != rhs->ee_zero_jacobian[i]) {
      return false;
    }
  }
  // ee_body_jacobian
  for (size_t i = 0; i < 42; ++i) {
    if (lhs->ee_body_jacobian[i] != rhs->ee_body_jacobian[i]) {
      return false;
    }
  }
  // mass
  for (size_t i = 0; i < 49; ++i) {
    if (lhs->mass[i] != rhs->mass[i]) {
      return false;
    }
  }
  return true;
}

bool
franka_msgs__msg__FrankaModel__copy(
  const franka_msgs__msg__FrankaModel * input,
  franka_msgs__msg__FrankaModel * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // coriolis
  for (size_t i = 0; i < 7; ++i) {
    output->coriolis[i] = input->coriolis[i];
  }
  // ee_zero_jacobian
  for (size_t i = 0; i < 42; ++i) {
    output->ee_zero_jacobian[i] = input->ee_zero_jacobian[i];
  }
  // ee_body_jacobian
  for (size_t i = 0; i < 42; ++i) {
    output->ee_body_jacobian[i] = input->ee_body_jacobian[i];
  }
  // mass
  for (size_t i = 0; i < 49; ++i) {
    output->mass[i] = input->mass[i];
  }
  return true;
}

franka_msgs__msg__FrankaModel *
franka_msgs__msg__FrankaModel__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  franka_msgs__msg__FrankaModel * msg = (franka_msgs__msg__FrankaModel *)allocator.allocate(sizeof(franka_msgs__msg__FrankaModel), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(franka_msgs__msg__FrankaModel));
  bool success = franka_msgs__msg__FrankaModel__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
franka_msgs__msg__FrankaModel__destroy(franka_msgs__msg__FrankaModel * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    franka_msgs__msg__FrankaModel__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
franka_msgs__msg__FrankaModel__Sequence__init(franka_msgs__msg__FrankaModel__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  franka_msgs__msg__FrankaModel * data = NULL;

  if (size) {
    data = (franka_msgs__msg__FrankaModel *)allocator.zero_allocate(size, sizeof(franka_msgs__msg__FrankaModel), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = franka_msgs__msg__FrankaModel__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        franka_msgs__msg__FrankaModel__fini(&data[i - 1]);
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
franka_msgs__msg__FrankaModel__Sequence__fini(franka_msgs__msg__FrankaModel__Sequence * array)
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
      franka_msgs__msg__FrankaModel__fini(&array->data[i]);
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

franka_msgs__msg__FrankaModel__Sequence *
franka_msgs__msg__FrankaModel__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  franka_msgs__msg__FrankaModel__Sequence * array = (franka_msgs__msg__FrankaModel__Sequence *)allocator.allocate(sizeof(franka_msgs__msg__FrankaModel__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = franka_msgs__msg__FrankaModel__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
franka_msgs__msg__FrankaModel__Sequence__destroy(franka_msgs__msg__FrankaModel__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    franka_msgs__msg__FrankaModel__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
franka_msgs__msg__FrankaModel__Sequence__are_equal(const franka_msgs__msg__FrankaModel__Sequence * lhs, const franka_msgs__msg__FrankaModel__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!franka_msgs__msg__FrankaModel__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
franka_msgs__msg__FrankaModel__Sequence__copy(
  const franka_msgs__msg__FrankaModel__Sequence * input,
  franka_msgs__msg__FrankaModel__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(franka_msgs__msg__FrankaModel);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    franka_msgs__msg__FrankaModel * data =
      (franka_msgs__msg__FrankaModel *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!franka_msgs__msg__FrankaModel__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          franka_msgs__msg__FrankaModel__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!franka_msgs__msg__FrankaModel__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
