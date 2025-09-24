// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from multi_mode_control_msgs:msg/ContactThresholds.idl
// generated code does not contain a copyright notice
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `tau_max`
// Member `tau_min`
#include "multi_mode_control_msgs/msg/detail/joint_array__functions.h"

bool
multi_mode_control_msgs__msg__ContactThresholds__init(multi_mode_control_msgs__msg__ContactThresholds * msg)
{
  if (!msg) {
    return false;
  }
  // f_max
  // f_min
  // tau_max
  if (!multi_mode_control_msgs__msg__JointArray__Sequence__init(&msg->tau_max, 0)) {
    multi_mode_control_msgs__msg__ContactThresholds__fini(msg);
    return false;
  }
  // tau_min
  if (!multi_mode_control_msgs__msg__JointArray__Sequence__init(&msg->tau_min, 0)) {
    multi_mode_control_msgs__msg__ContactThresholds__fini(msg);
    return false;
  }
  // f_abs_max
  // m_abs_max
  // abs_mode
  // directional_mode
  // torque_mode
  return true;
}

void
multi_mode_control_msgs__msg__ContactThresholds__fini(multi_mode_control_msgs__msg__ContactThresholds * msg)
{
  if (!msg) {
    return;
  }
  // f_max
  // f_min
  // tau_max
  multi_mode_control_msgs__msg__JointArray__Sequence__fini(&msg->tau_max);
  // tau_min
  multi_mode_control_msgs__msg__JointArray__Sequence__fini(&msg->tau_min);
  // f_abs_max
  // m_abs_max
  // abs_mode
  // directional_mode
  // torque_mode
}

bool
multi_mode_control_msgs__msg__ContactThresholds__are_equal(const multi_mode_control_msgs__msg__ContactThresholds * lhs, const multi_mode_control_msgs__msg__ContactThresholds * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // f_max
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_max[i] != rhs->f_max[i]) {
      return false;
    }
  }
  // f_min
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_min[i] != rhs->f_min[i]) {
      return false;
    }
  }
  // tau_max
  if (!multi_mode_control_msgs__msg__JointArray__Sequence__are_equal(
      &(lhs->tau_max), &(rhs->tau_max)))
  {
    return false;
  }
  // tau_min
  if (!multi_mode_control_msgs__msg__JointArray__Sequence__are_equal(
      &(lhs->tau_min), &(rhs->tau_min)))
  {
    return false;
  }
  // f_abs_max
  if (lhs->f_abs_max != rhs->f_abs_max) {
    return false;
  }
  // m_abs_max
  if (lhs->m_abs_max != rhs->m_abs_max) {
    return false;
  }
  // abs_mode
  if (lhs->abs_mode != rhs->abs_mode) {
    return false;
  }
  // directional_mode
  if (lhs->directional_mode != rhs->directional_mode) {
    return false;
  }
  // torque_mode
  if (lhs->torque_mode != rhs->torque_mode) {
    return false;
  }
  return true;
}

bool
multi_mode_control_msgs__msg__ContactThresholds__copy(
  const multi_mode_control_msgs__msg__ContactThresholds * input,
  multi_mode_control_msgs__msg__ContactThresholds * output)
{
  if (!input || !output) {
    return false;
  }
  // f_max
  for (size_t i = 0; i < 6; ++i) {
    output->f_max[i] = input->f_max[i];
  }
  // f_min
  for (size_t i = 0; i < 6; ++i) {
    output->f_min[i] = input->f_min[i];
  }
  // tau_max
  if (!multi_mode_control_msgs__msg__JointArray__Sequence__copy(
      &(input->tau_max), &(output->tau_max)))
  {
    return false;
  }
  // tau_min
  if (!multi_mode_control_msgs__msg__JointArray__Sequence__copy(
      &(input->tau_min), &(output->tau_min)))
  {
    return false;
  }
  // f_abs_max
  output->f_abs_max = input->f_abs_max;
  // m_abs_max
  output->m_abs_max = input->m_abs_max;
  // abs_mode
  output->abs_mode = input->abs_mode;
  // directional_mode
  output->directional_mode = input->directional_mode;
  // torque_mode
  output->torque_mode = input->torque_mode;
  return true;
}

multi_mode_control_msgs__msg__ContactThresholds *
multi_mode_control_msgs__msg__ContactThresholds__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__ContactThresholds * msg = (multi_mode_control_msgs__msg__ContactThresholds *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__ContactThresholds), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(multi_mode_control_msgs__msg__ContactThresholds));
  bool success = multi_mode_control_msgs__msg__ContactThresholds__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
multi_mode_control_msgs__msg__ContactThresholds__destroy(multi_mode_control_msgs__msg__ContactThresholds * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    multi_mode_control_msgs__msg__ContactThresholds__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
multi_mode_control_msgs__msg__ContactThresholds__Sequence__init(multi_mode_control_msgs__msg__ContactThresholds__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__ContactThresholds * data = NULL;

  if (size) {
    data = (multi_mode_control_msgs__msg__ContactThresholds *)allocator.zero_allocate(size, sizeof(multi_mode_control_msgs__msg__ContactThresholds), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = multi_mode_control_msgs__msg__ContactThresholds__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        multi_mode_control_msgs__msg__ContactThresholds__fini(&data[i - 1]);
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
multi_mode_control_msgs__msg__ContactThresholds__Sequence__fini(multi_mode_control_msgs__msg__ContactThresholds__Sequence * array)
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
      multi_mode_control_msgs__msg__ContactThresholds__fini(&array->data[i]);
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

multi_mode_control_msgs__msg__ContactThresholds__Sequence *
multi_mode_control_msgs__msg__ContactThresholds__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  multi_mode_control_msgs__msg__ContactThresholds__Sequence * array = (multi_mode_control_msgs__msg__ContactThresholds__Sequence *)allocator.allocate(sizeof(multi_mode_control_msgs__msg__ContactThresholds__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = multi_mode_control_msgs__msg__ContactThresholds__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
multi_mode_control_msgs__msg__ContactThresholds__Sequence__destroy(multi_mode_control_msgs__msg__ContactThresholds__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    multi_mode_control_msgs__msg__ContactThresholds__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
multi_mode_control_msgs__msg__ContactThresholds__Sequence__are_equal(const multi_mode_control_msgs__msg__ContactThresholds__Sequence * lhs, const multi_mode_control_msgs__msg__ContactThresholds__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!multi_mode_control_msgs__msg__ContactThresholds__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
multi_mode_control_msgs__msg__ContactThresholds__Sequence__copy(
  const multi_mode_control_msgs__msg__ContactThresholds__Sequence * input,
  multi_mode_control_msgs__msg__ContactThresholds__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(multi_mode_control_msgs__msg__ContactThresholds);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    multi_mode_control_msgs__msg__ContactThresholds * data =
      (multi_mode_control_msgs__msg__ContactThresholds *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!multi_mode_control_msgs__msg__ContactThresholds__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          multi_mode_control_msgs__msg__ContactThresholds__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!multi_mode_control_msgs__msg__ContactThresholds__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
