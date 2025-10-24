// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mujoco_ros_msgs:msg/PluginStats.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/plugin_stats__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `plugin_type`
#include "rosidl_runtime_c/string_functions.h"

bool
mujoco_ros_msgs__msg__PluginStats__init(mujoco_ros_msgs__msg__PluginStats * msg)
{
  if (!msg) {
    return false;
  }
  // plugin_type
  if (!rosidl_runtime_c__String__init(&msg->plugin_type)) {
    mujoco_ros_msgs__msg__PluginStats__fini(msg);
    return false;
  }
  // load_time
  // reset_time
  // ema_steptime_control
  // ema_steptime_passive
  // ema_steptime_render
  // ema_steptime_last_stage
  return true;
}

void
mujoco_ros_msgs__msg__PluginStats__fini(mujoco_ros_msgs__msg__PluginStats * msg)
{
  if (!msg) {
    return;
  }
  // plugin_type
  rosidl_runtime_c__String__fini(&msg->plugin_type);
  // load_time
  // reset_time
  // ema_steptime_control
  // ema_steptime_passive
  // ema_steptime_render
  // ema_steptime_last_stage
}

bool
mujoco_ros_msgs__msg__PluginStats__are_equal(const mujoco_ros_msgs__msg__PluginStats * lhs, const mujoco_ros_msgs__msg__PluginStats * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // plugin_type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->plugin_type), &(rhs->plugin_type)))
  {
    return false;
  }
  // load_time
  if (lhs->load_time != rhs->load_time) {
    return false;
  }
  // reset_time
  if (lhs->reset_time != rhs->reset_time) {
    return false;
  }
  // ema_steptime_control
  if (lhs->ema_steptime_control != rhs->ema_steptime_control) {
    return false;
  }
  // ema_steptime_passive
  if (lhs->ema_steptime_passive != rhs->ema_steptime_passive) {
    return false;
  }
  // ema_steptime_render
  if (lhs->ema_steptime_render != rhs->ema_steptime_render) {
    return false;
  }
  // ema_steptime_last_stage
  if (lhs->ema_steptime_last_stage != rhs->ema_steptime_last_stage) {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__msg__PluginStats__copy(
  const mujoco_ros_msgs__msg__PluginStats * input,
  mujoco_ros_msgs__msg__PluginStats * output)
{
  if (!input || !output) {
    return false;
  }
  // plugin_type
  if (!rosidl_runtime_c__String__copy(
      &(input->plugin_type), &(output->plugin_type)))
  {
    return false;
  }
  // load_time
  output->load_time = input->load_time;
  // reset_time
  output->reset_time = input->reset_time;
  // ema_steptime_control
  output->ema_steptime_control = input->ema_steptime_control;
  // ema_steptime_passive
  output->ema_steptime_passive = input->ema_steptime_passive;
  // ema_steptime_render
  output->ema_steptime_render = input->ema_steptime_render;
  // ema_steptime_last_stage
  output->ema_steptime_last_stage = input->ema_steptime_last_stage;
  return true;
}

mujoco_ros_msgs__msg__PluginStats *
mujoco_ros_msgs__msg__PluginStats__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__PluginStats * msg = (mujoco_ros_msgs__msg__PluginStats *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__PluginStats), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__msg__PluginStats));
  bool success = mujoco_ros_msgs__msg__PluginStats__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__msg__PluginStats__destroy(mujoco_ros_msgs__msg__PluginStats * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__msg__PluginStats__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__msg__PluginStats__Sequence__init(mujoco_ros_msgs__msg__PluginStats__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__PluginStats * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__msg__PluginStats *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__msg__PluginStats), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__msg__PluginStats__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__msg__PluginStats__fini(&data[i - 1]);
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
mujoco_ros_msgs__msg__PluginStats__Sequence__fini(mujoco_ros_msgs__msg__PluginStats__Sequence * array)
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
      mujoco_ros_msgs__msg__PluginStats__fini(&array->data[i]);
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

mujoco_ros_msgs__msg__PluginStats__Sequence *
mujoco_ros_msgs__msg__PluginStats__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__PluginStats__Sequence * array = (mujoco_ros_msgs__msg__PluginStats__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__PluginStats__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__msg__PluginStats__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__msg__PluginStats__Sequence__destroy(mujoco_ros_msgs__msg__PluginStats__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__msg__PluginStats__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__msg__PluginStats__Sequence__are_equal(const mujoco_ros_msgs__msg__PluginStats__Sequence * lhs, const mujoco_ros_msgs__msg__PluginStats__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__msg__PluginStats__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__msg__PluginStats__Sequence__copy(
  const mujoco_ros_msgs__msg__PluginStats__Sequence * input,
  mujoco_ros_msgs__msg__PluginStats__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__msg__PluginStats);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__msg__PluginStats * data =
      (mujoco_ros_msgs__msg__PluginStats *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__msg__PluginStats__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__msg__PluginStats__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__msg__PluginStats__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
