// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from mujoco_ros_msgs:msg/SolverParameters.idl
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
#include "mujoco_ros_msgs/msg/detail/solver_parameters__struct.h"
#include "mujoco_ros_msgs/msg/detail/solver_parameters__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool mujoco_ros_msgs__msg__solver_parameters__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[56];
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
    assert(strncmp("mujoco_ros_msgs.msg._solver_parameters.SolverParameters", full_classname_dest, 55) == 0);
  }
  mujoco_ros_msgs__msg__SolverParameters * ros_message = _ros_message;
  {  // dmin
    PyObject * field = PyObject_GetAttrString(_pymsg, "dmin");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->dmin = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // dmax
    PyObject * field = PyObject_GetAttrString(_pymsg, "dmax");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->dmax = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // width
    PyObject * field = PyObject_GetAttrString(_pymsg, "width");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->width = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // midpoint
    PyObject * field = PyObject_GetAttrString(_pymsg, "midpoint");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->midpoint = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // power
    PyObject * field = PyObject_GetAttrString(_pymsg, "power");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->power = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // timeconst
    PyObject * field = PyObject_GetAttrString(_pymsg, "timeconst");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->timeconst = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // dampratio
    PyObject * field = PyObject_GetAttrString(_pymsg, "dampratio");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->dampratio = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * mujoco_ros_msgs__msg__solver_parameters__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SolverParameters */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("mujoco_ros_msgs.msg._solver_parameters");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SolverParameters");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  mujoco_ros_msgs__msg__SolverParameters * ros_message = (mujoco_ros_msgs__msg__SolverParameters *)raw_ros_message;
  {  // dmin
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->dmin);
    {
      int rc = PyObject_SetAttrString(_pymessage, "dmin", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // dmax
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->dmax);
    {
      int rc = PyObject_SetAttrString(_pymessage, "dmax", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // width
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->width);
    {
      int rc = PyObject_SetAttrString(_pymessage, "width", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // midpoint
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->midpoint);
    {
      int rc = PyObject_SetAttrString(_pymessage, "midpoint", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // power
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->power);
    {
      int rc = PyObject_SetAttrString(_pymessage, "power", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // timeconst
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->timeconst);
    {
      int rc = PyObject_SetAttrString(_pymessage, "timeconst", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // dampratio
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->dampratio);
    {
      int rc = PyObject_SetAttrString(_pymessage, "dampratio", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
