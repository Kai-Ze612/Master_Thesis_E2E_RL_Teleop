// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from mujoco_ros_msgs:msg/GeomProperties.idl
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
#include "mujoco_ros_msgs/msg/detail/geom_properties__struct.h"
#include "mujoco_ros_msgs/msg/detail/geom_properties__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

bool mujoco_ros_msgs__msg__geom_type__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * mujoco_ros_msgs__msg__geom_type__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool mujoco_ros_msgs__msg__geom_properties__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[52];
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
    assert(strncmp("mujoco_ros_msgs.msg._geom_properties.GeomProperties", full_classname_dest, 51) == 0);
  }
  mujoco_ros_msgs__msg__GeomProperties * ros_message = _ros_message;
  {  // name
    PyObject * field = PyObject_GetAttrString(_pymsg, "name");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->name, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // type
    PyObject * field = PyObject_GetAttrString(_pymsg, "type");
    if (!field) {
      return false;
    }
    if (!mujoco_ros_msgs__msg__geom_type__convert_from_py(field, &ros_message->type)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // body_mass
    PyObject * field = PyObject_GetAttrString(_pymsg, "body_mass");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->body_mass = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // size_0
    PyObject * field = PyObject_GetAttrString(_pymsg, "size_0");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->size_0 = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // size_1
    PyObject * field = PyObject_GetAttrString(_pymsg, "size_1");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->size_1 = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // size_2
    PyObject * field = PyObject_GetAttrString(_pymsg, "size_2");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->size_2 = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // friction_slide
    PyObject * field = PyObject_GetAttrString(_pymsg, "friction_slide");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->friction_slide = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // friction_spin
    PyObject * field = PyObject_GetAttrString(_pymsg, "friction_spin");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->friction_spin = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // friction_roll
    PyObject * field = PyObject_GetAttrString(_pymsg, "friction_roll");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->friction_roll = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * mujoco_ros_msgs__msg__geom_properties__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of GeomProperties */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("mujoco_ros_msgs.msg._geom_properties");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "GeomProperties");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  mujoco_ros_msgs__msg__GeomProperties * ros_message = (mujoco_ros_msgs__msg__GeomProperties *)raw_ros_message;
  {  // name
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->name.data,
      strlen(ros_message->name.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "name", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // type
    PyObject * field = NULL;
    field = mujoco_ros_msgs__msg__geom_type__convert_to_py(&ros_message->type);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "type", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // body_mass
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->body_mass);
    {
      int rc = PyObject_SetAttrString(_pymessage, "body_mass", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // size_0
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->size_0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "size_0", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // size_1
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->size_1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "size_1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // size_2
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->size_2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "size_2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // friction_slide
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->friction_slide);
    {
      int rc = PyObject_SetAttrString(_pymessage, "friction_slide", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // friction_spin
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->friction_spin);
    {
      int rc = PyObject_SetAttrString(_pymessage, "friction_spin", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // friction_roll
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->friction_roll);
    {
      int rc = PyObject_SetAttrString(_pymessage, "friction_roll", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
