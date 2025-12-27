// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from panda_motion_generator_msgs:action/JointViaMotion.idl
// generated code does not contain a copyright notice
#include "panda_motion_generator_msgs/action/detail/joint_via_motion__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `via_poses`
#include "panda_motion_generator_msgs/msg/detail/joint_pose__functions.h"

bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__init(panda_motion_generator_msgs__action__JointViaMotion_Goal * msg)
{
  if (!msg) {
    return false;
  }
  // via_poses
  if (!panda_motion_generator_msgs__msg__JointPose__Sequence__init(&msg->via_poses, 0)) {
    panda_motion_generator_msgs__action__JointViaMotion_Goal__fini(msg);
    return false;
  }
  // v_scale
  return true;
}

void
panda_motion_generator_msgs__action__JointViaMotion_Goal__fini(panda_motion_generator_msgs__action__JointViaMotion_Goal * msg)
{
  if (!msg) {
    return;
  }
  // via_poses
  panda_motion_generator_msgs__msg__JointPose__Sequence__fini(&msg->via_poses);
  // v_scale
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Goal * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Goal * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // via_poses
  if (!panda_motion_generator_msgs__msg__JointPose__Sequence__are_equal(
      &(lhs->via_poses), &(rhs->via_poses)))
  {
    return false;
  }
  // v_scale
  if (lhs->v_scale != rhs->v_scale) {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Goal * input,
  panda_motion_generator_msgs__action__JointViaMotion_Goal * output)
{
  if (!input || !output) {
    return false;
  }
  // via_poses
  if (!panda_motion_generator_msgs__msg__JointPose__Sequence__copy(
      &(input->via_poses), &(output->via_poses)))
  {
    return false;
  }
  // v_scale
  output->v_scale = input->v_scale;
  return true;
}

panda_motion_generator_msgs__action__JointViaMotion_Goal *
panda_motion_generator_msgs__action__JointViaMotion_Goal__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_Goal * msg = (panda_motion_generator_msgs__action__JointViaMotion_Goal *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_Goal), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__action__JointViaMotion_Goal));
  bool success = panda_motion_generator_msgs__action__JointViaMotion_Goal__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__action__JointViaMotion_Goal__destroy(panda_motion_generator_msgs__action__JointViaMotion_Goal * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__action__JointViaMotion_Goal__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_Goal * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__action__JointViaMotion_Goal *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__action__JointViaMotion_Goal), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__action__JointViaMotion_Goal__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__action__JointViaMotion_Goal__fini(&data[i - 1]);
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
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * array)
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
      panda_motion_generator_msgs__action__JointViaMotion_Goal__fini(&array->data[i]);
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

panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * array = (panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_Goal__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_Goal__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__action__JointViaMotion_Goal);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__action__JointViaMotion_Goal * data =
      (panda_motion_generator_msgs__action__JointViaMotion_Goal *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__action__JointViaMotion_Goal__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__action__JointViaMotion_Goal__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_Goal__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `result`
#include "panda_motion_generator_msgs/msg/detail/simple_action_result__functions.h"

bool
panda_motion_generator_msgs__action__JointViaMotion_Result__init(panda_motion_generator_msgs__action__JointViaMotion_Result * msg)
{
  if (!msg) {
    return false;
  }
  // result
  if (!panda_motion_generator_msgs__msg__SimpleActionResult__init(&msg->result)) {
    panda_motion_generator_msgs__action__JointViaMotion_Result__fini(msg);
    return false;
  }
  return true;
}

void
panda_motion_generator_msgs__action__JointViaMotion_Result__fini(panda_motion_generator_msgs__action__JointViaMotion_Result * msg)
{
  if (!msg) {
    return;
  }
  // result
  panda_motion_generator_msgs__msg__SimpleActionResult__fini(&msg->result);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Result__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Result * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Result * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // result
  if (!panda_motion_generator_msgs__msg__SimpleActionResult__are_equal(
      &(lhs->result), &(rhs->result)))
  {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Result__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Result * input,
  panda_motion_generator_msgs__action__JointViaMotion_Result * output)
{
  if (!input || !output) {
    return false;
  }
  // result
  if (!panda_motion_generator_msgs__msg__SimpleActionResult__copy(
      &(input->result), &(output->result)))
  {
    return false;
  }
  return true;
}

panda_motion_generator_msgs__action__JointViaMotion_Result *
panda_motion_generator_msgs__action__JointViaMotion_Result__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_Result * msg = (panda_motion_generator_msgs__action__JointViaMotion_Result *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_Result), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__action__JointViaMotion_Result));
  bool success = panda_motion_generator_msgs__action__JointViaMotion_Result__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__action__JointViaMotion_Result__destroy(panda_motion_generator_msgs__action__JointViaMotion_Result * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__action__JointViaMotion_Result__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_Result * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__action__JointViaMotion_Result *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__action__JointViaMotion_Result), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__action__JointViaMotion_Result__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__action__JointViaMotion_Result__fini(&data[i - 1]);
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
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * array)
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
      panda_motion_generator_msgs__action__JointViaMotion_Result__fini(&array->data[i]);
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

panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * array = (panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_Result__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_Result__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__action__JointViaMotion_Result);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__action__JointViaMotion_Result * data =
      (panda_motion_generator_msgs__action__JointViaMotion_Result *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__action__JointViaMotion_Result__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__action__JointViaMotion_Result__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_Result__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__init(panda_motion_generator_msgs__action__JointViaMotion_Feedback * msg)
{
  if (!msg) {
    return false;
  }
  // progress
  // time_to_completion
  return true;
}

void
panda_motion_generator_msgs__action__JointViaMotion_Feedback__fini(panda_motion_generator_msgs__action__JointViaMotion_Feedback * msg)
{
  if (!msg) {
    return;
  }
  // progress
  // time_to_completion
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Feedback * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Feedback * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // progress
  if (lhs->progress != rhs->progress) {
    return false;
  }
  // time_to_completion
  if (lhs->time_to_completion != rhs->time_to_completion) {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Feedback * input,
  panda_motion_generator_msgs__action__JointViaMotion_Feedback * output)
{
  if (!input || !output) {
    return false;
  }
  // progress
  output->progress = input->progress;
  // time_to_completion
  output->time_to_completion = input->time_to_completion;
  return true;
}

panda_motion_generator_msgs__action__JointViaMotion_Feedback *
panda_motion_generator_msgs__action__JointViaMotion_Feedback__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_Feedback * msg = (panda_motion_generator_msgs__action__JointViaMotion_Feedback *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_Feedback), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__action__JointViaMotion_Feedback));
  bool success = panda_motion_generator_msgs__action__JointViaMotion_Feedback__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__action__JointViaMotion_Feedback__destroy(panda_motion_generator_msgs__action__JointViaMotion_Feedback * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__action__JointViaMotion_Feedback__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_Feedback * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__action__JointViaMotion_Feedback *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__action__JointViaMotion_Feedback), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__action__JointViaMotion_Feedback__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__action__JointViaMotion_Feedback__fini(&data[i - 1]);
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
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * array)
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
      panda_motion_generator_msgs__action__JointViaMotion_Feedback__fini(&array->data[i]);
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

panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * array = (panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_Feedback__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_Feedback__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__action__JointViaMotion_Feedback);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__action__JointViaMotion_Feedback * data =
      (panda_motion_generator_msgs__action__JointViaMotion_Feedback *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__action__JointViaMotion_Feedback__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__action__JointViaMotion_Feedback__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_Feedback__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
#include "unique_identifier_msgs/msg/detail/uuid__functions.h"
// Member `goal`
// already included above
// #include "panda_motion_generator_msgs/action/detail/joint_via_motion__functions.h"

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__init(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini(msg);
    return false;
  }
  // goal
  if (!panda_motion_generator_msgs__action__JointViaMotion_Goal__init(&msg->goal)) {
    panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini(msg);
    return false;
  }
  return true;
}

void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // goal
  panda_motion_generator_msgs__action__JointViaMotion_Goal__fini(&msg->goal);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * lhs, const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  // goal
  if (!panda_motion_generator_msgs__action__JointViaMotion_Goal__are_equal(
      &(lhs->goal), &(rhs->goal)))
  {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * input,
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  // goal
  if (!panda_motion_generator_msgs__action__JointViaMotion_Goal__copy(
      &(input->goal), &(output->goal)))
  {
    return false;
  }
  return true;
}

panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request *
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * msg = (panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request));
  bool success = panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__destroy(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini(&data[i - 1]);
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
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * array)
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
      panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini(&array->data[i]);
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

panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * array = (panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request * data =
      (panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__init(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * msg)
{
  if (!msg) {
    return false;
  }
  // accepted
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__fini(msg);
    return false;
  }
  return true;
}

void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__fini(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * msg)
{
  if (!msg) {
    return;
  }
  // accepted
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * lhs, const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // accepted
  if (lhs->accepted != rhs->accepted) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * input,
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // accepted
  output->accepted = input->accepted;
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  return true;
}

panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response *
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * msg = (panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response));
  bool success = panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__destroy(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__fini(&data[i - 1]);
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
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * array)
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
      panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__fini(&array->data[i]);
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

panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * array = (panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response * data =
      (panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_SendGoal_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__functions.h"

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__init(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__fini(msg);
    return false;
  }
  return true;
}

void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__fini(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * lhs, const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * input,
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  return true;
}

panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request *
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * msg = (panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request));
  bool success = panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__destroy(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__fini(&data[i - 1]);
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
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * array)
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
      panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__fini(&array->data[i]);
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

panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * array = (panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request * data =
      (panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_GetResult_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `result`
// already included above
// #include "panda_motion_generator_msgs/action/detail/joint_via_motion__functions.h"

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__init(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * msg)
{
  if (!msg) {
    return false;
  }
  // status
  // result
  if (!panda_motion_generator_msgs__action__JointViaMotion_Result__init(&msg->result)) {
    panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__fini(msg);
    return false;
  }
  return true;
}

void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__fini(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * msg)
{
  if (!msg) {
    return;
  }
  // status
  // result
  panda_motion_generator_msgs__action__JointViaMotion_Result__fini(&msg->result);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * lhs, const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // status
  if (lhs->status != rhs->status) {
    return false;
  }
  // result
  if (!panda_motion_generator_msgs__action__JointViaMotion_Result__are_equal(
      &(lhs->result), &(rhs->result)))
  {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * input,
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // status
  output->status = input->status;
  // result
  if (!panda_motion_generator_msgs__action__JointViaMotion_Result__copy(
      &(input->result), &(output->result)))
  {
    return false;
  }
  return true;
}

panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response *
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * msg = (panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response));
  bool success = panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__destroy(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__fini(&data[i - 1]);
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
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * array)
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
      panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__fini(&array->data[i]);
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

panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * array = (panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response * data =
      (panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_GetResult_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__functions.h"
// Member `feedback`
// already included above
// #include "panda_motion_generator_msgs/action/detail/joint_via_motion__functions.h"

bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__init(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini(msg);
    return false;
  }
  // feedback
  if (!panda_motion_generator_msgs__action__JointViaMotion_Feedback__init(&msg->feedback)) {
    panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini(msg);
    return false;
  }
  return true;
}

void
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // feedback
  panda_motion_generator_msgs__action__JointViaMotion_Feedback__fini(&msg->feedback);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * lhs, const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  // feedback
  if (!panda_motion_generator_msgs__action__JointViaMotion_Feedback__are_equal(
      &(lhs->feedback), &(rhs->feedback)))
  {
    return false;
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * input,
  panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  // feedback
  if (!panda_motion_generator_msgs__action__JointViaMotion_Feedback__copy(
      &(input->feedback), &(output->feedback)))
  {
    return false;
  }
  return true;
}

panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage *
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * msg = (panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage));
  bool success = panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__destroy(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__init(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * data = NULL;

  if (size) {
    data = (panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage *)allocator.zero_allocate(size, sizeof(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini(&data[i - 1]);
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
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__fini(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * array)
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
      panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini(&array->data[i]);
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

panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence *
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * array = (panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence *)allocator.allocate(sizeof(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__destroy(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__are_equal(const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * lhs, const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence__copy(
  const panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * input,
  panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage * data =
      (panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!panda_motion_generator_msgs__action__JointViaMotion_FeedbackMessage__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
