// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mujoco_ros_msgs:msg/GeomProperties.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/msg/detail/geom_properties__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `name`
#include "rosidl_runtime_c/string_functions.h"
// Member `type`
#include "mujoco_ros_msgs/msg/detail/geom_type__functions.h"

bool
mujoco_ros_msgs__msg__GeomProperties__init(mujoco_ros_msgs__msg__GeomProperties * msg)
{
  if (!msg) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__init(&msg->name)) {
    mujoco_ros_msgs__msg__GeomProperties__fini(msg);
    return false;
  }
  // type
  if (!mujoco_ros_msgs__msg__GeomType__init(&msg->type)) {
    mujoco_ros_msgs__msg__GeomProperties__fini(msg);
    return false;
  }
  // body_mass
  // size_0
  // size_1
  // size_2
  // friction_slide
  // friction_spin
  // friction_roll
  return true;
}

void
mujoco_ros_msgs__msg__GeomProperties__fini(mujoco_ros_msgs__msg__GeomProperties * msg)
{
  if (!msg) {
    return;
  }
  // name
  rosidl_runtime_c__String__fini(&msg->name);
  // type
  mujoco_ros_msgs__msg__GeomType__fini(&msg->type);
  // body_mass
  // size_0
  // size_1
  // size_2
  // friction_slide
  // friction_spin
  // friction_roll
}

bool
mujoco_ros_msgs__msg__GeomProperties__are_equal(const mujoco_ros_msgs__msg__GeomProperties * lhs, const mujoco_ros_msgs__msg__GeomProperties * rhs)
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
  // type
  if (!mujoco_ros_msgs__msg__GeomType__are_equal(
      &(lhs->type), &(rhs->type)))
  {
    return false;
  }
  // body_mass
  if (lhs->body_mass != rhs->body_mass) {
    return false;
  }
  // size_0
  if (lhs->size_0 != rhs->size_0) {
    return false;
  }
  // size_1
  if (lhs->size_1 != rhs->size_1) {
    return false;
  }
  // size_2
  if (lhs->size_2 != rhs->size_2) {
    return false;
  }
  // friction_slide
  if (lhs->friction_slide != rhs->friction_slide) {
    return false;
  }
  // friction_spin
  if (lhs->friction_spin != rhs->friction_spin) {
    return false;
  }
  // friction_roll
  if (lhs->friction_roll != rhs->friction_roll) {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__msg__GeomProperties__copy(
  const mujoco_ros_msgs__msg__GeomProperties * input,
  mujoco_ros_msgs__msg__GeomProperties * output)
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
  // type
  if (!mujoco_ros_msgs__msg__GeomType__copy(
      &(input->type), &(output->type)))
  {
    return false;
  }
  // body_mass
  output->body_mass = input->body_mass;
  // size_0
  output->size_0 = input->size_0;
  // size_1
  output->size_1 = input->size_1;
  // size_2
  output->size_2 = input->size_2;
  // friction_slide
  output->friction_slide = input->friction_slide;
  // friction_spin
  output->friction_spin = input->friction_spin;
  // friction_roll
  output->friction_roll = input->friction_roll;
  return true;
}

mujoco_ros_msgs__msg__GeomProperties *
mujoco_ros_msgs__msg__GeomProperties__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__GeomProperties * msg = (mujoco_ros_msgs__msg__GeomProperties *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__GeomProperties), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__msg__GeomProperties));
  bool success = mujoco_ros_msgs__msg__GeomProperties__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__msg__GeomProperties__destroy(mujoco_ros_msgs__msg__GeomProperties * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__msg__GeomProperties__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__msg__GeomProperties__Sequence__init(mujoco_ros_msgs__msg__GeomProperties__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__GeomProperties * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__msg__GeomProperties *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__msg__GeomProperties), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__msg__GeomProperties__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__msg__GeomProperties__fini(&data[i - 1]);
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
mujoco_ros_msgs__msg__GeomProperties__Sequence__fini(mujoco_ros_msgs__msg__GeomProperties__Sequence * array)
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
      mujoco_ros_msgs__msg__GeomProperties__fini(&array->data[i]);
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

mujoco_ros_msgs__msg__GeomProperties__Sequence *
mujoco_ros_msgs__msg__GeomProperties__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__msg__GeomProperties__Sequence * array = (mujoco_ros_msgs__msg__GeomProperties__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__msg__GeomProperties__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__msg__GeomProperties__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__msg__GeomProperties__Sequence__destroy(mujoco_ros_msgs__msg__GeomProperties__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__msg__GeomProperties__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__msg__GeomProperties__Sequence__are_equal(const mujoco_ros_msgs__msg__GeomProperties__Sequence * lhs, const mujoco_ros_msgs__msg__GeomProperties__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__msg__GeomProperties__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__msg__GeomProperties__Sequence__copy(
  const mujoco_ros_msgs__msg__GeomProperties__Sequence * input,
  mujoco_ros_msgs__msg__GeomProperties__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__msg__GeomProperties);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__msg__GeomProperties * data =
      (mujoco_ros_msgs__msg__GeomProperties *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__msg__GeomProperties__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__msg__GeomProperties__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__msg__GeomProperties__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
