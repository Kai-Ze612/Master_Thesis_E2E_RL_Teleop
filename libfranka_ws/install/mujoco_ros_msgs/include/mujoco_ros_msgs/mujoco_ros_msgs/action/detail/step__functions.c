// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mujoco_ros_msgs:action/Step.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/action/detail/step__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
mujoco_ros_msgs__action__Step_Goal__init(mujoco_ros_msgs__action__Step_Goal * msg)
{
  if (!msg) {
    return false;
  }
  // num_steps
  return true;
}

void
mujoco_ros_msgs__action__Step_Goal__fini(mujoco_ros_msgs__action__Step_Goal * msg)
{
  if (!msg) {
    return;
  }
  // num_steps
}

bool
mujoco_ros_msgs__action__Step_Goal__are_equal(const mujoco_ros_msgs__action__Step_Goal * lhs, const mujoco_ros_msgs__action__Step_Goal * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // num_steps
  if (lhs->num_steps != rhs->num_steps) {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_Goal__copy(
  const mujoco_ros_msgs__action__Step_Goal * input,
  mujoco_ros_msgs__action__Step_Goal * output)
{
  if (!input || !output) {
    return false;
  }
  // num_steps
  output->num_steps = input->num_steps;
  return true;
}

mujoco_ros_msgs__action__Step_Goal *
mujoco_ros_msgs__action__Step_Goal__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_Goal * msg = (mujoco_ros_msgs__action__Step_Goal *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_Goal), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__action__Step_Goal));
  bool success = mujoco_ros_msgs__action__Step_Goal__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__action__Step_Goal__destroy(mujoco_ros_msgs__action__Step_Goal * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__action__Step_Goal__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__action__Step_Goal__Sequence__init(mujoco_ros_msgs__action__Step_Goal__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_Goal * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__action__Step_Goal *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__action__Step_Goal), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__action__Step_Goal__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__action__Step_Goal__fini(&data[i - 1]);
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
mujoco_ros_msgs__action__Step_Goal__Sequence__fini(mujoco_ros_msgs__action__Step_Goal__Sequence * array)
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
      mujoco_ros_msgs__action__Step_Goal__fini(&array->data[i]);
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

mujoco_ros_msgs__action__Step_Goal__Sequence *
mujoco_ros_msgs__action__Step_Goal__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_Goal__Sequence * array = (mujoco_ros_msgs__action__Step_Goal__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_Goal__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__action__Step_Goal__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__action__Step_Goal__Sequence__destroy(mujoco_ros_msgs__action__Step_Goal__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__action__Step_Goal__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__action__Step_Goal__Sequence__are_equal(const mujoco_ros_msgs__action__Step_Goal__Sequence * lhs, const mujoco_ros_msgs__action__Step_Goal__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_Goal__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_Goal__Sequence__copy(
  const mujoco_ros_msgs__action__Step_Goal__Sequence * input,
  mujoco_ros_msgs__action__Step_Goal__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__action__Step_Goal);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__action__Step_Goal * data =
      (mujoco_ros_msgs__action__Step_Goal *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__action__Step_Goal__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__action__Step_Goal__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_Goal__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
mujoco_ros_msgs__action__Step_Result__init(mujoco_ros_msgs__action__Step_Result * msg)
{
  if (!msg) {
    return false;
  }
  // success
  return true;
}

void
mujoco_ros_msgs__action__Step_Result__fini(mujoco_ros_msgs__action__Step_Result * msg)
{
  if (!msg) {
    return;
  }
  // success
}

bool
mujoco_ros_msgs__action__Step_Result__are_equal(const mujoco_ros_msgs__action__Step_Result * lhs, const mujoco_ros_msgs__action__Step_Result * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_Result__copy(
  const mujoco_ros_msgs__action__Step_Result * input,
  mujoco_ros_msgs__action__Step_Result * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  return true;
}

mujoco_ros_msgs__action__Step_Result *
mujoco_ros_msgs__action__Step_Result__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_Result * msg = (mujoco_ros_msgs__action__Step_Result *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_Result), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__action__Step_Result));
  bool success = mujoco_ros_msgs__action__Step_Result__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__action__Step_Result__destroy(mujoco_ros_msgs__action__Step_Result * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__action__Step_Result__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__action__Step_Result__Sequence__init(mujoco_ros_msgs__action__Step_Result__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_Result * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__action__Step_Result *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__action__Step_Result), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__action__Step_Result__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__action__Step_Result__fini(&data[i - 1]);
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
mujoco_ros_msgs__action__Step_Result__Sequence__fini(mujoco_ros_msgs__action__Step_Result__Sequence * array)
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
      mujoco_ros_msgs__action__Step_Result__fini(&array->data[i]);
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

mujoco_ros_msgs__action__Step_Result__Sequence *
mujoco_ros_msgs__action__Step_Result__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_Result__Sequence * array = (mujoco_ros_msgs__action__Step_Result__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_Result__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__action__Step_Result__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__action__Step_Result__Sequence__destroy(mujoco_ros_msgs__action__Step_Result__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__action__Step_Result__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__action__Step_Result__Sequence__are_equal(const mujoco_ros_msgs__action__Step_Result__Sequence * lhs, const mujoco_ros_msgs__action__Step_Result__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_Result__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_Result__Sequence__copy(
  const mujoco_ros_msgs__action__Step_Result__Sequence * input,
  mujoco_ros_msgs__action__Step_Result__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__action__Step_Result);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__action__Step_Result * data =
      (mujoco_ros_msgs__action__Step_Result *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__action__Step_Result__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__action__Step_Result__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_Result__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
mujoco_ros_msgs__action__Step_Feedback__init(mujoco_ros_msgs__action__Step_Feedback * msg)
{
  if (!msg) {
    return false;
  }
  // steps_left
  return true;
}

void
mujoco_ros_msgs__action__Step_Feedback__fini(mujoco_ros_msgs__action__Step_Feedback * msg)
{
  if (!msg) {
    return;
  }
  // steps_left
}

bool
mujoco_ros_msgs__action__Step_Feedback__are_equal(const mujoco_ros_msgs__action__Step_Feedback * lhs, const mujoco_ros_msgs__action__Step_Feedback * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // steps_left
  if (lhs->steps_left != rhs->steps_left) {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_Feedback__copy(
  const mujoco_ros_msgs__action__Step_Feedback * input,
  mujoco_ros_msgs__action__Step_Feedback * output)
{
  if (!input || !output) {
    return false;
  }
  // steps_left
  output->steps_left = input->steps_left;
  return true;
}

mujoco_ros_msgs__action__Step_Feedback *
mujoco_ros_msgs__action__Step_Feedback__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_Feedback * msg = (mujoco_ros_msgs__action__Step_Feedback *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_Feedback), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__action__Step_Feedback));
  bool success = mujoco_ros_msgs__action__Step_Feedback__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__action__Step_Feedback__destroy(mujoco_ros_msgs__action__Step_Feedback * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__action__Step_Feedback__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__action__Step_Feedback__Sequence__init(mujoco_ros_msgs__action__Step_Feedback__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_Feedback * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__action__Step_Feedback *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__action__Step_Feedback), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__action__Step_Feedback__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__action__Step_Feedback__fini(&data[i - 1]);
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
mujoco_ros_msgs__action__Step_Feedback__Sequence__fini(mujoco_ros_msgs__action__Step_Feedback__Sequence * array)
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
      mujoco_ros_msgs__action__Step_Feedback__fini(&array->data[i]);
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

mujoco_ros_msgs__action__Step_Feedback__Sequence *
mujoco_ros_msgs__action__Step_Feedback__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_Feedback__Sequence * array = (mujoco_ros_msgs__action__Step_Feedback__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_Feedback__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__action__Step_Feedback__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__action__Step_Feedback__Sequence__destroy(mujoco_ros_msgs__action__Step_Feedback__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__action__Step_Feedback__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__action__Step_Feedback__Sequence__are_equal(const mujoco_ros_msgs__action__Step_Feedback__Sequence * lhs, const mujoco_ros_msgs__action__Step_Feedback__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_Feedback__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_Feedback__Sequence__copy(
  const mujoco_ros_msgs__action__Step_Feedback__Sequence * input,
  mujoco_ros_msgs__action__Step_Feedback__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__action__Step_Feedback);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__action__Step_Feedback * data =
      (mujoco_ros_msgs__action__Step_Feedback *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__action__Step_Feedback__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__action__Step_Feedback__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_Feedback__copy(
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
// #include "mujoco_ros_msgs/action/detail/step__functions.h"

bool
mujoco_ros_msgs__action__Step_SendGoal_Request__init(mujoco_ros_msgs__action__Step_SendGoal_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    mujoco_ros_msgs__action__Step_SendGoal_Request__fini(msg);
    return false;
  }
  // goal
  if (!mujoco_ros_msgs__action__Step_Goal__init(&msg->goal)) {
    mujoco_ros_msgs__action__Step_SendGoal_Request__fini(msg);
    return false;
  }
  return true;
}

void
mujoco_ros_msgs__action__Step_SendGoal_Request__fini(mujoco_ros_msgs__action__Step_SendGoal_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // goal
  mujoco_ros_msgs__action__Step_Goal__fini(&msg->goal);
}

bool
mujoco_ros_msgs__action__Step_SendGoal_Request__are_equal(const mujoco_ros_msgs__action__Step_SendGoal_Request * lhs, const mujoco_ros_msgs__action__Step_SendGoal_Request * rhs)
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
  if (!mujoco_ros_msgs__action__Step_Goal__are_equal(
      &(lhs->goal), &(rhs->goal)))
  {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_SendGoal_Request__copy(
  const mujoco_ros_msgs__action__Step_SendGoal_Request * input,
  mujoco_ros_msgs__action__Step_SendGoal_Request * output)
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
  if (!mujoco_ros_msgs__action__Step_Goal__copy(
      &(input->goal), &(output->goal)))
  {
    return false;
  }
  return true;
}

mujoco_ros_msgs__action__Step_SendGoal_Request *
mujoco_ros_msgs__action__Step_SendGoal_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_SendGoal_Request * msg = (mujoco_ros_msgs__action__Step_SendGoal_Request *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_SendGoal_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__action__Step_SendGoal_Request));
  bool success = mujoco_ros_msgs__action__Step_SendGoal_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__action__Step_SendGoal_Request__destroy(mujoco_ros_msgs__action__Step_SendGoal_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__action__Step_SendGoal_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__init(mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_SendGoal_Request * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__action__Step_SendGoal_Request *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__action__Step_SendGoal_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__action__Step_SendGoal_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__action__Step_SendGoal_Request__fini(&data[i - 1]);
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
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__fini(mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * array)
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
      mujoco_ros_msgs__action__Step_SendGoal_Request__fini(&array->data[i]);
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

mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence *
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * array = (mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__destroy(mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__are_equal(const mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * lhs, const mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_SendGoal_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence__copy(
  const mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * input,
  mujoco_ros_msgs__action__Step_SendGoal_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__action__Step_SendGoal_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__action__Step_SendGoal_Request * data =
      (mujoco_ros_msgs__action__Step_SendGoal_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__action__Step_SendGoal_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__action__Step_SendGoal_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_SendGoal_Request__copy(
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
mujoco_ros_msgs__action__Step_SendGoal_Response__init(mujoco_ros_msgs__action__Step_SendGoal_Response * msg)
{
  if (!msg) {
    return false;
  }
  // accepted
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    mujoco_ros_msgs__action__Step_SendGoal_Response__fini(msg);
    return false;
  }
  return true;
}

void
mujoco_ros_msgs__action__Step_SendGoal_Response__fini(mujoco_ros_msgs__action__Step_SendGoal_Response * msg)
{
  if (!msg) {
    return;
  }
  // accepted
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
}

bool
mujoco_ros_msgs__action__Step_SendGoal_Response__are_equal(const mujoco_ros_msgs__action__Step_SendGoal_Response * lhs, const mujoco_ros_msgs__action__Step_SendGoal_Response * rhs)
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
mujoco_ros_msgs__action__Step_SendGoal_Response__copy(
  const mujoco_ros_msgs__action__Step_SendGoal_Response * input,
  mujoco_ros_msgs__action__Step_SendGoal_Response * output)
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

mujoco_ros_msgs__action__Step_SendGoal_Response *
mujoco_ros_msgs__action__Step_SendGoal_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_SendGoal_Response * msg = (mujoco_ros_msgs__action__Step_SendGoal_Response *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_SendGoal_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__action__Step_SendGoal_Response));
  bool success = mujoco_ros_msgs__action__Step_SendGoal_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__action__Step_SendGoal_Response__destroy(mujoco_ros_msgs__action__Step_SendGoal_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__action__Step_SendGoal_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__init(mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_SendGoal_Response * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__action__Step_SendGoal_Response *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__action__Step_SendGoal_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__action__Step_SendGoal_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__action__Step_SendGoal_Response__fini(&data[i - 1]);
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
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__fini(mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * array)
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
      mujoco_ros_msgs__action__Step_SendGoal_Response__fini(&array->data[i]);
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

mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence *
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * array = (mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__destroy(mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__are_equal(const mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * lhs, const mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_SendGoal_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence__copy(
  const mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * input,
  mujoco_ros_msgs__action__Step_SendGoal_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__action__Step_SendGoal_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__action__Step_SendGoal_Response * data =
      (mujoco_ros_msgs__action__Step_SendGoal_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__action__Step_SendGoal_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__action__Step_SendGoal_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_SendGoal_Response__copy(
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
mujoco_ros_msgs__action__Step_GetResult_Request__init(mujoco_ros_msgs__action__Step_GetResult_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    mujoco_ros_msgs__action__Step_GetResult_Request__fini(msg);
    return false;
  }
  return true;
}

void
mujoco_ros_msgs__action__Step_GetResult_Request__fini(mujoco_ros_msgs__action__Step_GetResult_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
}

bool
mujoco_ros_msgs__action__Step_GetResult_Request__are_equal(const mujoco_ros_msgs__action__Step_GetResult_Request * lhs, const mujoco_ros_msgs__action__Step_GetResult_Request * rhs)
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
mujoco_ros_msgs__action__Step_GetResult_Request__copy(
  const mujoco_ros_msgs__action__Step_GetResult_Request * input,
  mujoco_ros_msgs__action__Step_GetResult_Request * output)
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

mujoco_ros_msgs__action__Step_GetResult_Request *
mujoco_ros_msgs__action__Step_GetResult_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_GetResult_Request * msg = (mujoco_ros_msgs__action__Step_GetResult_Request *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_GetResult_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__action__Step_GetResult_Request));
  bool success = mujoco_ros_msgs__action__Step_GetResult_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__action__Step_GetResult_Request__destroy(mujoco_ros_msgs__action__Step_GetResult_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__action__Step_GetResult_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__init(mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_GetResult_Request * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__action__Step_GetResult_Request *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__action__Step_GetResult_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__action__Step_GetResult_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__action__Step_GetResult_Request__fini(&data[i - 1]);
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
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__fini(mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * array)
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
      mujoco_ros_msgs__action__Step_GetResult_Request__fini(&array->data[i]);
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

mujoco_ros_msgs__action__Step_GetResult_Request__Sequence *
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * array = (mujoco_ros_msgs__action__Step_GetResult_Request__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_GetResult_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__destroy(mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__are_equal(const mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * lhs, const mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_GetResult_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_GetResult_Request__Sequence__copy(
  const mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * input,
  mujoco_ros_msgs__action__Step_GetResult_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__action__Step_GetResult_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__action__Step_GetResult_Request * data =
      (mujoco_ros_msgs__action__Step_GetResult_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__action__Step_GetResult_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__action__Step_GetResult_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_GetResult_Request__copy(
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
// #include "mujoco_ros_msgs/action/detail/step__functions.h"

bool
mujoco_ros_msgs__action__Step_GetResult_Response__init(mujoco_ros_msgs__action__Step_GetResult_Response * msg)
{
  if (!msg) {
    return false;
  }
  // status
  // result
  if (!mujoco_ros_msgs__action__Step_Result__init(&msg->result)) {
    mujoco_ros_msgs__action__Step_GetResult_Response__fini(msg);
    return false;
  }
  return true;
}

void
mujoco_ros_msgs__action__Step_GetResult_Response__fini(mujoco_ros_msgs__action__Step_GetResult_Response * msg)
{
  if (!msg) {
    return;
  }
  // status
  // result
  mujoco_ros_msgs__action__Step_Result__fini(&msg->result);
}

bool
mujoco_ros_msgs__action__Step_GetResult_Response__are_equal(const mujoco_ros_msgs__action__Step_GetResult_Response * lhs, const mujoco_ros_msgs__action__Step_GetResult_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // status
  if (lhs->status != rhs->status) {
    return false;
  }
  // result
  if (!mujoco_ros_msgs__action__Step_Result__are_equal(
      &(lhs->result), &(rhs->result)))
  {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_GetResult_Response__copy(
  const mujoco_ros_msgs__action__Step_GetResult_Response * input,
  mujoco_ros_msgs__action__Step_GetResult_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // status
  output->status = input->status;
  // result
  if (!mujoco_ros_msgs__action__Step_Result__copy(
      &(input->result), &(output->result)))
  {
    return false;
  }
  return true;
}

mujoco_ros_msgs__action__Step_GetResult_Response *
mujoco_ros_msgs__action__Step_GetResult_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_GetResult_Response * msg = (mujoco_ros_msgs__action__Step_GetResult_Response *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_GetResult_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__action__Step_GetResult_Response));
  bool success = mujoco_ros_msgs__action__Step_GetResult_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__action__Step_GetResult_Response__destroy(mujoco_ros_msgs__action__Step_GetResult_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__action__Step_GetResult_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__init(mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_GetResult_Response * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__action__Step_GetResult_Response *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__action__Step_GetResult_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__action__Step_GetResult_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__action__Step_GetResult_Response__fini(&data[i - 1]);
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
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__fini(mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * array)
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
      mujoco_ros_msgs__action__Step_GetResult_Response__fini(&array->data[i]);
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

mujoco_ros_msgs__action__Step_GetResult_Response__Sequence *
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * array = (mujoco_ros_msgs__action__Step_GetResult_Response__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_GetResult_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__destroy(mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__are_equal(const mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * lhs, const mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_GetResult_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_GetResult_Response__Sequence__copy(
  const mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * input,
  mujoco_ros_msgs__action__Step_GetResult_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__action__Step_GetResult_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__action__Step_GetResult_Response * data =
      (mujoco_ros_msgs__action__Step_GetResult_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__action__Step_GetResult_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__action__Step_GetResult_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_GetResult_Response__copy(
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
// #include "mujoco_ros_msgs/action/detail/step__functions.h"

bool
mujoco_ros_msgs__action__Step_FeedbackMessage__init(mujoco_ros_msgs__action__Step_FeedbackMessage * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    mujoco_ros_msgs__action__Step_FeedbackMessage__fini(msg);
    return false;
  }
  // feedback
  if (!mujoco_ros_msgs__action__Step_Feedback__init(&msg->feedback)) {
    mujoco_ros_msgs__action__Step_FeedbackMessage__fini(msg);
    return false;
  }
  return true;
}

void
mujoco_ros_msgs__action__Step_FeedbackMessage__fini(mujoco_ros_msgs__action__Step_FeedbackMessage * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // feedback
  mujoco_ros_msgs__action__Step_Feedback__fini(&msg->feedback);
}

bool
mujoco_ros_msgs__action__Step_FeedbackMessage__are_equal(const mujoco_ros_msgs__action__Step_FeedbackMessage * lhs, const mujoco_ros_msgs__action__Step_FeedbackMessage * rhs)
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
  if (!mujoco_ros_msgs__action__Step_Feedback__are_equal(
      &(lhs->feedback), &(rhs->feedback)))
  {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_FeedbackMessage__copy(
  const mujoco_ros_msgs__action__Step_FeedbackMessage * input,
  mujoco_ros_msgs__action__Step_FeedbackMessage * output)
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
  if (!mujoco_ros_msgs__action__Step_Feedback__copy(
      &(input->feedback), &(output->feedback)))
  {
    return false;
  }
  return true;
}

mujoco_ros_msgs__action__Step_FeedbackMessage *
mujoco_ros_msgs__action__Step_FeedbackMessage__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_FeedbackMessage * msg = (mujoco_ros_msgs__action__Step_FeedbackMessage *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_FeedbackMessage), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__action__Step_FeedbackMessage));
  bool success = mujoco_ros_msgs__action__Step_FeedbackMessage__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__action__Step_FeedbackMessage__destroy(mujoco_ros_msgs__action__Step_FeedbackMessage * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__action__Step_FeedbackMessage__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__init(mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_FeedbackMessage * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__action__Step_FeedbackMessage *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__action__Step_FeedbackMessage), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__action__Step_FeedbackMessage__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__action__Step_FeedbackMessage__fini(&data[i - 1]);
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
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__fini(mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * array)
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
      mujoco_ros_msgs__action__Step_FeedbackMessage__fini(&array->data[i]);
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

mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence *
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * array = (mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__destroy(mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__are_equal(const mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * lhs, const mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_FeedbackMessage__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence__copy(
  const mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * input,
  mujoco_ros_msgs__action__Step_FeedbackMessage__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__action__Step_FeedbackMessage);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__action__Step_FeedbackMessage * data =
      (mujoco_ros_msgs__action__Step_FeedbackMessage *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__action__Step_FeedbackMessage__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__action__Step_FeedbackMessage__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__action__Step_FeedbackMessage__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
