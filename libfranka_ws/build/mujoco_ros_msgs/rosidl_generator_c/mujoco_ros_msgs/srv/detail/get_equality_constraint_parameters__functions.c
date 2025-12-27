// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mujoco_ros_msgs:srv/GetEqualityConstraintParameters.idl
// generated code does not contain a copyright notice
#include "mujoco_ros_msgs/srv/detail/get_equality_constraint_parameters__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `names`
// Member `admin_hash`
#include "rosidl_runtime_c/string_functions.h"

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__init(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * msg)
{
  if (!msg) {
    return false;
  }
  // names
  if (!rosidl_runtime_c__String__Sequence__init(&msg->names, 0)) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__fini(msg);
    return false;
  }
  // admin_hash
  if (!rosidl_runtime_c__String__init(&msg->admin_hash)) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__fini(msg);
    return false;
  }
  return true;
}

void
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__fini(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * msg)
{
  if (!msg) {
    return;
  }
  // names
  rosidl_runtime_c__String__Sequence__fini(&msg->names);
  // admin_hash
  rosidl_runtime_c__String__fini(&msg->admin_hash);
}

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__are_equal(const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * lhs, const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // names
  if (!rosidl_runtime_c__String__Sequence__are_equal(
      &(lhs->names), &(rhs->names)))
  {
    return false;
  }
  // admin_hash
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->admin_hash), &(rhs->admin_hash)))
  {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__copy(
  const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * input,
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // names
  if (!rosidl_runtime_c__String__Sequence__copy(
      &(input->names), &(output->names)))
  {
    return false;
  }
  // admin_hash
  if (!rosidl_runtime_c__String__copy(
      &(input->admin_hash), &(output->admin_hash)))
  {
    return false;
  }
  return true;
}

mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request *
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * msg = (mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request *)allocator.allocate(sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request));
  bool success = mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__destroy(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence__init(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__fini(&data[i - 1]);
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
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence__fini(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence * array)
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
      mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__fini(&array->data[i]);
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

mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence *
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence * array = (mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence__destroy(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence__are_equal(const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence * lhs, const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence__copy(
  const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence * input,
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request * data =
      (mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `parameters`
#include "mujoco_ros_msgs/msg/detail/equality_constraint_parameters__functions.h"
// Member `status_message`
// already included above
// #include "rosidl_runtime_c/string_functions.h"

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__init(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * msg)
{
  if (!msg) {
    return false;
  }
  // parameters
  if (!mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__init(&msg->parameters, 0)) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__fini(msg);
    return false;
  }
  // success
  // status_message
  if (!rosidl_runtime_c__String__init(&msg->status_message)) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__fini(msg);
    return false;
  }
  return true;
}

void
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__fini(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * msg)
{
  if (!msg) {
    return;
  }
  // parameters
  mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__fini(&msg->parameters);
  // success
  // status_message
  rosidl_runtime_c__String__fini(&msg->status_message);
}

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__are_equal(const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * lhs, const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // parameters
  if (!mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__are_equal(
      &(lhs->parameters), &(rhs->parameters)))
  {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // status_message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->status_message), &(rhs->status_message)))
  {
    return false;
  }
  return true;
}

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__copy(
  const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * input,
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // parameters
  if (!mujoco_ros_msgs__msg__EqualityConstraintParameters__Sequence__copy(
      &(input->parameters), &(output->parameters)))
  {
    return false;
  }
  // success
  output->success = input->success;
  // status_message
  if (!rosidl_runtime_c__String__copy(
      &(input->status_message), &(output->status_message)))
  {
    return false;
  }
  return true;
}

mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response *
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * msg = (mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response *)allocator.allocate(sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response));
  bool success = mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__destroy(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence__init(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * data = NULL;

  if (size) {
    data = (mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response *)allocator.zero_allocate(size, sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__fini(&data[i - 1]);
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
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence__fini(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence * array)
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
      mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__fini(&array->data[i]);
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

mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence *
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence * array = (mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence *)allocator.allocate(sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence__destroy(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence__are_equal(const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence * lhs, const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence__copy(
  const mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence * input,
  mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response * data =
      (mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mujoco_ros_msgs__srv__GetEqualityConstraintParameters_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
