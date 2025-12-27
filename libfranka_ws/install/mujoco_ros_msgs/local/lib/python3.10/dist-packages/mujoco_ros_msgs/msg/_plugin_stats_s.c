// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from mujoco_ros_msgs:msg/PluginStats.idl
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
#include "mujoco_ros_msgs/msg/detail/plugin_stats__struct.h"
#include "mujoco_ros_msgs/msg/detail/plugin_stats__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool mujoco_ros_msgs__msg__plugin_stats__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[46];
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
    assert(strncmp("mujoco_ros_msgs.msg._plugin_stats.PluginStats", full_classname_dest, 45) == 0);
  }
  mujoco_ros_msgs__msg__PluginStats * ros_message = _ros_message;
  {  // plugin_type
    PyObject * field = PyObject_GetAttrString(_pymsg, "plugin_type");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->plugin_type, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // load_time
    PyObject * field = PyObject_GetAttrString(_pymsg, "load_time");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->load_time = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // reset_time
    PyObject * field = PyObject_GetAttrString(_pymsg, "reset_time");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->reset_time = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ema_steptime_control
    PyObject * field = PyObject_GetAttrString(_pymsg, "ema_steptime_control");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ema_steptime_control = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ema_steptime_passive
    PyObject * field = PyObject_GetAttrString(_pymsg, "ema_steptime_passive");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ema_steptime_passive = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ema_steptime_render
    PyObject * field = PyObject_GetAttrString(_pymsg, "ema_steptime_render");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ema_steptime_render = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ema_steptime_last_stage
    PyObject * field = PyObject_GetAttrString(_pymsg, "ema_steptime_last_stage");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ema_steptime_last_stage = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * mujoco_ros_msgs__msg__plugin_stats__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PluginStats */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("mujoco_ros_msgs.msg._plugin_stats");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PluginStats");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  mujoco_ros_msgs__msg__PluginStats * ros_message = (mujoco_ros_msgs__msg__PluginStats *)raw_ros_message;
  {  // plugin_type
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->plugin_type.data,
      strlen(ros_message->plugin_type.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "plugin_type", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // load_time
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->load_time);
    {
      int rc = PyObject_SetAttrString(_pymessage, "load_time", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // reset_time
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->reset_time);
    {
      int rc = PyObject_SetAttrString(_pymessage, "reset_time", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ema_steptime_control
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ema_steptime_control);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ema_steptime_control", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ema_steptime_passive
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ema_steptime_passive);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ema_steptime_passive", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ema_steptime_render
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ema_steptime_render);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ema_steptime_render", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ema_steptime_last_stage
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ema_steptime_last_stage);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ema_steptime_last_stage", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
