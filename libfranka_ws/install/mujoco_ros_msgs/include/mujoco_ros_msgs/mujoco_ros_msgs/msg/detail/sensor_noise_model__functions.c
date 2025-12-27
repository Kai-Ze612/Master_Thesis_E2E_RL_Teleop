// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mujoco_ros_msgs:msg/SensorNoiseModel.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/sensor_noise_model__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `sensor_name`
#include "rosidl_runtime_c/string_functions.h"
// Member `mean`
// Member `std`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
mujoco_ros_msgs__msg__SensorNoiseModel__init(mujoco_ros_msgs__msg__SensorNoiseModel * msg)
{
  if (!msg) {
    return false;
  }
  // sensor_name
  if (!rosidl_runtime_c__String__init(&msg->sensor_name)) {
    mujoco_ros_msgs__msg__SensorNoiseModel__fini(msg);
    return false;
  }
  // mean
  if (!rosidl_runtime_c__double__Sequence__init(&msg->mean, 0)) {
    mujoco_ros_msgs__msg__SensorNoiseModel__fini(msg);
    return false;
  }
  // std
  if (!rosidl_runtime_c__double__Sequence__init(&msg->std, 0)) {
    mujoco_ros_msgs__msg__SensorNoiseModel__fini(msg);
    return false;
  }
  // set_flag
  return true;
}

void
mujoco_ros_msgs__msg__SensorNoiseModel__fini(mujoco_ros_msgs__msg__SensorNoiseModel * msg)
{
  if (!msg) {
    return;
  }
  // sensor_name
  rosidl_runtime_c__String__fini(&msg->sensor_name);
  // mean
  rosidl_runtime_c__double__Sequence__fini(&msg->mean);
  // std
  rosidl_runtime_c__double__Sequence__fini(&msg->std);
  // set_flag
}

bool
mujoco_ros_msgs__msg__SensorNoiseModel__are_equal(const mujoco_ros_msgs__msg__SensorNoiseModel * lhs, const mujoco_ros_msgs__msg__SensorNoiseModel * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // sensor_name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->sensor_name), &(rhs->sensor_name)))
  {
    return false;
  }
  // mean
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->mean), &(rhs->mean)))
  {
    return false;
  }
  // std
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->std), &(rhs->std)))
  {
    return false;
  }
  // set_flag
  if (lhs->set_flag != rhs->set_flag) {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__msg__SensorNoiseModel__copy(
  const mujoco_ros_msgs__msg__SensorNoiseModel * input,
  mujoco_ros_msgs__msg__SensorNoiseModel * output)
{
  if (!input || !output) {
    return false;
  }
  // sensor_name
  if (!rosidl_runtime_c__String__copy(
      &(input->sensor_name), &(output->sensor_name)))
  {
    return false;
  }
  // mean
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->mean), &(output->mean)))
  {
    return false;
  }
  // std
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->std), &(output->std)))
  {
    return false;
  }
  // set_flag
  output->set_flag = input->set_flag;
  return true;
}

mujoco_ros_msgs__msg__SensorNoiseModel *
mujoco_ros_msgs__msg__SensorNoiseModel__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__SensorNoiseModel * msg = (mujoco_ros_msgs__msg__SensorNoiseModel *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__SensorNoiseModel), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__msg__SensorNoiseModel));
  bool success = mujoco_ros_msgs__msg__SensorNoiseModel__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__msg__SensorNoiseModel__destroy(mujoco_ros_msgs__msg__SensorNoiseModel * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__msg__SensorNoiseModel__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__msg__SensorNoiseModel__Sequence__init(mujoco_ros_msgs__msg__SensorNoiseModel__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__SensorNoiseModel * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__msg__SensorNoiseModel *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__msg__SensorNoiseModel), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__msg__SensorNoiseModel__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__msg__SensorNoiseModel__fini(&data[i - 1]);
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
mujoco_ros_msgs__msg__SensorNoiseModel__Sequence__fini(mujoco_ros_msgs__msg__SensorNoiseModel__Sequence * array)
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
      mujoco_ros_msgs__msg__SensorNoiseModel__fini(&array->data[i]);
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

mujoco_ros_msgs__msg__SensorNoiseModel__Sequence *
mujoco_ros_msgs__msg__SensorNoiseModel__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__SensorNoiseModel__Sequence * array = (mujoco_ros_msgs__msg__SensorNoiseModel__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__SensorNoiseModel__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__msg__SensorNoiseModel__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__msg__SensorNoiseModel__Sequence__destroy(mujoco_ros_msgs__msg__SensorNoiseModel__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__msg__SensorNoiseModel__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__msg__SensorNoiseModel__Sequence__are_equal(const mujoco_ros_msgs__msg__SensorNoiseModel__Sequence * lhs, const mujoco_ros_msgs__msg__SensorNoiseModel__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__msg__SensorNoiseModel__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__msg__SensorNoiseModel__Sequence__copy(
  const mujoco_ros_msgs__msg__SensorNoiseModel__Sequence * input,
  mujoco_ros_msgs__msg__SensorNoiseModel__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__msg__SensorNoiseModel);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__msg__SensorNoiseModel * data =
      (mujoco_ros_msgs__msg__SensorNoiseModel *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__msg__SensorNoiseModel__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__msg__SensorNoiseModel__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__msg__SensorNoiseModel__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
