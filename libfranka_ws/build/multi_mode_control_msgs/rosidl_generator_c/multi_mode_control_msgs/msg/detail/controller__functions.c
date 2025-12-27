// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from multi_mode_control_msgs:msg/Controller.idl
// generated code does not contain a copyright notice
#include "multi_mode_control_msgs/msg/detail/controller__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `name`
// Member `resources`
#include "rosidl_runtime_c/string_functions.h"

bool
multi_mode_control_msgs__msg__Controller__init(multi_mode_control_msgs__msg__Controller * msg)
{
  if (!msg) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__init(&msg->name)) {
    multi_mode_control_msgs__msg__Controller__fini(msg);
    return false;
  }
  // resources
  if (!rosidl_runtime_c__String__Sequence__init(&msg->resources, 0)) {
    multi_mode_control_msgs__msg__Controller__fini(msg);
    return false;
  }
  return true;
}

void
multi_mode_control_msgs__msg__Controller__fini(multi_mode_control_msgs__msg__Controller * msg)
{
  if (!msg) {
    return;
  }
  // name
  rosidl_runtime_c__String__fini(&msg->name);
  // resources
  rosidl_runtime_c__String__Sequence__fini(&msg->resources);
}

bool
multi_mode_control_msgs__msg__Controller__are_equal(const multi_mode_control_msgs__msg__Controller * lhs, const multi_mode_control_msgs__msg__Controller * rhs)
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
  // resources
  if (!rosidl_runtime_c__String__Sequence__are_equal(
      &(lhs->resources), &(rhs->resources)))
  {
    return false;
  }
  return true;
}

bool
multi_mode_control_msgs__msg__Controller__copy(
  const multi_mode_control_msgs__msg__Controller * input,
  multi_mode_control_msgs__msg__Controller * output)
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
  // resources
  if (!rosidl_runtime_c__String__Sequence__copy(
      &(input->resources), &(output->resources)))
  {
    return false;
  }
  return true;
}

multi_mode_control_msgs__msg__Controller *
multi_mode_control_msgs__msg__Controller__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__Controller * msg = (multi_mode_control_msgs__msg__Controller *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__Controller), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(multi_mode_control_msgs__msg__Controller));
  bool success = multi_mode_control_msgs__msg__Controller__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
multi_mode_control_msgs__msg__Controller__destroy(multi_mode_control_msgs__msg__Controller * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    multi_mode_control_msgs__msg__Controller__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
multi_mode_control_msgs__msg__Controller__Sequence__init(multi_mode_control_msgs__msg__Controller__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__Controller * data = NULL;

  if (size) {
    data = (multi_mode_control_msgs__msg__Controller *)allocator.zero_allocate(size, sizeof(multi_mode_control_msgs__msg__Controller), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = multi_mode_control_msgs__msg__Controller__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        multi_mode_control_msgs__msg__Controller__fini(&data[i - 1]);
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
multi_mode_control_msgs__msg__Controller__Sequence__fini(multi_mode_control_msgs__msg__Controller__Sequence * array)
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
      multi_mode_control_msgs__msg__Controller__fini(&array->data[i]);
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

multi_mode_control_msgs__msg__Controller__Sequence *
multi_mode_control_msgs__msg__Controller__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__Controller__Sequence * array = (multi_mode_control_msgs__msg__Controller__Sequence *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__Controller__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = multi_mode_control_msgs__msg__Controller__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
multi_mode_control_msgs__msg__Controller__Sequence__destroy(multi_mode_control_msgs__msg__Controller__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    multi_mode_control_msgs__msg__Controller__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
multi_mode_control_msgs__msg__Controller__Sequence__are_equal(const multi_mode_control_msgs__msg__Controller__Sequence * lhs, const multi_mode_control_msgs__msg__Controller__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!multi_mode_control_msgs__msg__Controller__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
multi_mode_control_msgs__msg__Controller__Sequence__copy(
  const multi_mode_control_msgs__msg__Controller__Sequence * input,
  multi_mode_control_msgs__msg__Controller__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(multi_mode_control_msgs__msg__Controller);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    multi_mode_control_msgs__msg__Controller * data =
      (multi_mode_control_msgs__msg__Controller *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!multi_mode_control_msgs__msg__Controller__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          multi_mode_control_msgs__msg__Controller__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!multi_mode_control_msgs__msg__Controller__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
