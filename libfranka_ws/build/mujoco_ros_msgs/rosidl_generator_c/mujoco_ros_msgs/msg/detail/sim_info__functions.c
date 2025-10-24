// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mujoco_ros_msgs:msg/SimInfo.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/sim_info__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `model_path`
#include "rosidl_runtime_c/string_functions.h"
// Member `loading_state`
#include "mujoco_ros_msgs/msg/detail/state_uint__functions.h"

bool
mujoco_ros_msgs__msg__SimInfo__init(mujoco_ros_msgs__msg__SimInfo * msg)
{
  if (!msg) {
    return false;
  }
  // model_path
  if (!rosidl_runtime_c__String__init(&msg->model_path)) {
    mujoco_ros_msgs__msg__SimInfo__fini(msg);
    return false;
  }
  // model_valid
  // load_count
  // loading_state
  if (!mujoco_ros_msgs__msg__StateUint__init(&msg->loading_state)) {
    mujoco_ros_msgs__msg__SimInfo__fini(msg);
    return false;
  }
  // paused
  // pending_sim_steps
  // rt_measured
  // rt_setting
  return true;
}

void
mujoco_ros_msgs__msg__SimInfo__fini(mujoco_ros_msgs__msg__SimInfo * msg)
{
  if (!msg) {
    return;
  }
  // model_path
  rosidl_runtime_c__String__fini(&msg->model_path);
  // model_valid
  // load_count
  // loading_state
  mujoco_ros_msgs__msg__StateUint__fini(&msg->loading_state);
  // paused
  // pending_sim_steps
  // rt_measured
  // rt_setting
}

bool
mujoco_ros_msgs__msg__SimInfo__are_equal(const mujoco_ros_msgs__msg__SimInfo * lhs, const mujoco_ros_msgs__msg__SimInfo * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // model_path
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->model_path), &(rhs->model_path)))
  {
    return false;
  }
  // model_valid
  if (lhs->model_valid != rhs->model_valid) {
    return false;
  }
  // load_count
  if (lhs->load_count != rhs->load_count) {
    return false;
  }
  // loading_state
  if (!mujoco_ros_msgs__msg__StateUint__are_equal(
      &(lhs->loading_state), &(rhs->loading_state)))
  {
    return false;
  }
  // paused
  if (lhs->paused != rhs->paused) {
    return false;
  }
  // pending_sim_steps
  if (lhs->pending_sim_steps != rhs->pending_sim_steps) {
    return false;
  }
  // rt_measured
  if (lhs->rt_measured != rhs->rt_measured) {
    return false;
  }
  // rt_setting
  if (lhs->rt_setting != rhs->rt_setting) {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__msg__SimInfo__copy(
  const mujoco_ros_msgs__msg__SimInfo * input,
  mujoco_ros_msgs__msg__SimInfo * output)
{
  if (!input || !output) {
    return false;
  }
  // model_path
  if (!rosidl_runtime_c__String__copy(
      &(input->model_path), &(output->model_path)))
  {
    return false;
  }
  // model_valid
  output->model_valid = input->model_valid;
  // load_count
  output->load_count = input->load_count;
  // loading_state
  if (!mujoco_ros_msgs__msg__StateUint__copy(
      &(input->loading_state), &(output->loading_state)))
  {
    return false;
  }
  // paused
  output->paused = input->paused;
  // pending_sim_steps
  output->pending_sim_steps = input->pending_sim_steps;
  // rt_measured
  output->rt_measured = input->rt_measured;
  // rt_setting
  output->rt_setting = input->rt_setting;
  return true;
}

mujoco_ros_msgs__msg__SimInfo *
mujoco_ros_msgs__msg__SimInfo__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__SimInfo * msg = (mujoco_ros_msgs__msg__SimInfo *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__SimInfo), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__msg__SimInfo));
  bool success = mujoco_ros_msgs__msg__SimInfo__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__msg__SimInfo__destroy(mujoco_ros_msgs__msg__SimInfo * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__msg__SimInfo__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__msg__SimInfo__Sequence__init(mujoco_ros_msgs__msg__SimInfo__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__SimInfo * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__msg__SimInfo *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__msg__SimInfo), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__msg__SimInfo__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__msg__SimInfo__fini(&data[i - 1]);
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
mujoco_ros_msgs__msg__SimInfo__Sequence__fini(mujoco_ros_msgs__msg__SimInfo__Sequence * array)
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
      mujoco_ros_msgs__msg__SimInfo__fini(&array->data[i]);
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

mujoco_ros_msgs__msg__SimInfo__Sequence *
mujoco_ros_msgs__msg__SimInfo__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__SimInfo__Sequence * array = (mujoco_ros_msgs__msg__SimInfo__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__SimInfo__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__msg__SimInfo__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__msg__SimInfo__Sequence__destroy(mujoco_ros_msgs__msg__SimInfo__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__msg__SimInfo__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__msg__SimInfo__Sequence__are_equal(const mujoco_ros_msgs__msg__SimInfo__Sequence * lhs, const mujoco_ros_msgs__msg__SimInfo__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__msg__SimInfo__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__msg__SimInfo__Sequence__copy(
  const mujoco_ros_msgs__msg__SimInfo__Sequence * input,
  mujoco_ros_msgs__msg__SimInfo__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__msg__SimInfo);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__msg__SimInfo * data =
      (mujoco_ros_msgs__msg__SimInfo *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__msg__SimInfo__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__msg__SimInfo__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__msg__SimInfo__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
