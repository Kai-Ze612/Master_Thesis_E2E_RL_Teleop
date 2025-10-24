// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from mujoco_ros_msgs:srv/SetBodyState.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "mujoco_ros_msgs/srv/detail/set_body_state__struct.h"
#include "mujoco_ros_msgs/srv/detail/set_body_state__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

bool mujoco_ros_msgs__msg__body_state__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * mujoco_ros_msgs__msg__body_state__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool mujoco_ros_msgs__srv__set_body_state__request__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[57];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("mujoco_ros_msgs.srv._set_body_state.SetBodyState_Request", full_classname_dest, 56) == 0);
  }
  mujoco_ros_msgs__srv__SetBodyState_Request * ros_message = _ros_message;
  {  // state
    PyObject * field = PyObject_GetAttrString(_pymsg, "state");
    if (!field) {
      return false;
    }
    if (!mujoco_ros_msgs__msg__body_state__convert_from_py(field, &ros_message->state)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // set_pose
    PyObject * field = PyObject_GetAttrString(_pymsg, "set_pose");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->set_pose = (Py_True == field);
    Py_DECREF(field);
  }
  {  // set_twist
    PyObject * field = PyObject_GetAttrString(_pymsg, "set_twist");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->set_twist = (Py_True == field);
    Py_DECREF(field);
  }
  {  // set_mass
    PyObject * field = PyObject_GetAttrString(_pymsg, "set_mass");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->set_mass = (Py_True == field);
    Py_DECREF(field);
  }
  {  // reset_qpos
    PyObject * field = PyObject_GetAttrString(_pymsg, "reset_qpos");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->reset_qpos = (Py_True == field);
    Py_DECREF(field);
  }
  {  // admin_hash
    PyObject * field = PyObject_GetAttrString(_pymsg, "admin_hash");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->admin_hash, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * mujoco_ros_msgs__srv__set_body_state__request__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SetBodyState_Request */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("mujoco_ros_msgs.srv._set_body_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SetBodyState_Request");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  mujoco_ros_msgs__srv__SetBodyState_Request * ros_message = (mujoco_ros_msgs__srv__SetBodyState_Request *)raw_ros_message;
  {  // state
    PyObject * field = NULL;
    field = mujoco_ros_msgs__msg__body_state__convert_to_py(&ros_message->state);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // set_pose
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->set_pose ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "set_pose", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // set_twist
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->set_twist ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "set_twist", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // set_mass
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->set_mass ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "set_mass", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // reset_qpos
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->reset_qpos ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "reset_qpos", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // admin_hash
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->admin_hash.data,
      strlen(ros_message->admin_hash.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "admin_hash", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/set_body_state__struct.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/set_body_state__functions.h"

// already included above
// #include "rosidl_runtime_c/string.h"
// already included above
// #include "rosidl_runtime_c/string_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool mujoco_ros_msgs__srv__set_body_state__response__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[58];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("mujoco_ros_msgs.srv._set_body_state.SetBodyState_Response", full_classname_dest, 57) == 0);
  }
  mujoco_ros_msgs__srv__SetBodyState_Response * ros_message = _ros_message;
  {  // success
    PyObject * field = PyObject_GetAttrString(_pymsg, "success");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->success = (Py_True == field);
    Py_DECREF(field);
  }
  {  // status_message
    PyObject * field = PyObject_GetAttrString(_pymsg, "status_message");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->status_message, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * mujoco_ros_msgs__srv__set_body_state__response__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SetBodyState_Response */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("mujoco_ros_msgs.srv._set_body_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SetBodyState_Response");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  mujoco_ros_msgs__srv__SetBodyState_Response * ros_message = (mujoco_ros_msgs__srv__SetBodyState_Response *)raw_ros_message;
  {  // success
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->success ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "success", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // status_message
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->status_message.data,
      strlen(ros_message->status_message.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "status_message", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
