// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from mujoco_ros_msgs:msg/SimInfo.idl
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
#include "mujoco_ros_msgs/msg/detail/sim_info__struct.h"
#include "mujoco_ros_msgs/msg/detail/sim_info__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

bool mujoco_ros_msgs__msg__state_uint__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * mujoco_ros_msgs__msg__state_uint__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool mujoco_ros_msgs__msg__sim_info__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[38];
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
    assert(strncmp("mujoco_ros_msgs.msg._sim_info.SimInfo", full_classname_dest, 37) == 0);
  }
  mujoco_ros_msgs__msg__SimInfo * ros_message = _ros_message;
  {  // model_path
    PyObject * field = PyObject_GetAttrString(_pymsg, "model_path");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->model_path, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // model_valid
    PyObject * field = PyObject_GetAttrString(_pymsg, "model_valid");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->model_valid = (Py_True == field);
    Py_DECREF(field);
  }
  {  // load_count
    PyObject * field = PyObject_GetAttrString(_pymsg, "load_count");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->load_count = (uint16_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // loading_state
    PyObject * field = PyObject_GetAttrString(_pymsg, "loading_state");
    if (!field) {
      return false;
    }
    if (!mujoco_ros_msgs__msg__state_uint__convert_from_py(field, &ros_message->loading_state)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // paused
    PyObject * field = PyObject_GetAttrString(_pymsg, "paused");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->paused = (Py_True == field);
    Py_DECREF(field);
  }
  {  // pending_sim_steps
    PyObject * field = PyObject_GetAttrString(_pymsg, "pending_sim_steps");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->pending_sim_steps = (uint16_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // rt_measured
    PyObject * field = PyObject_GetAttrString(_pymsg, "rt_measured");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rt_measured = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // rt_setting
    PyObject * field = PyObject_GetAttrString(_pymsg, "rt_setting");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rt_setting = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * mujoco_ros_msgs__msg__sim_info__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SimInfo */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("mujoco_ros_msgs.msg._sim_info");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SimInfo");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  mujoco_ros_msgs__msg__SimInfo * ros_message = (mujoco_ros_msgs__msg__SimInfo *)raw_ros_message;
  {  // model_path
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->model_path.data,
      strlen(ros_message->model_path.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "model_path", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // model_valid
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->model_valid ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "model_valid", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // load_count
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->load_count);
    {
      int rc = PyObject_SetAttrString(_pymessage, "load_count", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // loading_state
    PyObject * field = NULL;
    field = mujoco_ros_msgs__msg__state_uint__convert_to_py(&ros_message->loading_state);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "loading_state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // paused
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->paused ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "paused", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pending_sim_steps
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->pending_sim_steps);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pending_sim_steps", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rt_measured
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rt_measured);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rt_measured", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rt_setting
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rt_setting);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rt_setting", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
