// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from multi_mode_control_msgs:msg/ContactThresholds.idl
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
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__struct.h"
#include "multi_mode_control_msgs/msg/detail/contact_thresholds__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"

// Nested array functions includes
#include "multi_mode_control_msgs/msg/detail/joint_array__functions.h"
// end nested array functions include
bool multi_mode_control_msgs__msg__joint_array__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * multi_mode_control_msgs__msg__joint_array__convert_to_py(void * raw_ros_message);
bool multi_mode_control_msgs__msg__joint_array__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * multi_mode_control_msgs__msg__joint_array__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool multi_mode_control_msgs__msg__contact_thresholds__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[66];
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
    assert(strncmp("multi_mode_control_msgs.msg._contact_thresholds.ContactThresholds", full_classname_dest, 65) == 0);
  }
  multi_mode_control_msgs__msg__ContactThresholds * ros_message = _ros_message;
  {  // f_max
    PyObject * field = PyObject_GetAttrString(_pymsg, "f_max");
    if (!field) {
      return false;
    }
    {
      // TODO(dirk-thomas) use a better way to check the type before casting
      assert(field->ob_type != NULL);
      assert(field->ob_type->tp_name != NULL);
      assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
      PyArrayObject * seq_field = (PyArrayObject *)field;
      Py_INCREF(seq_field);
      assert(PyArray_NDIM(seq_field) == 1);
      assert(PyArray_TYPE(seq_field) == NPY_FLOAT64);
      Py_ssize_t size = 6;
      double * dest = ros_message->f_max;
      for (Py_ssize_t i = 0; i < size; ++i) {
        double tmp = *(npy_float64 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(double));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // f_min
    PyObject * field = PyObject_GetAttrString(_pymsg, "f_min");
    if (!field) {
      return false;
    }
    {
      // TODO(dirk-thomas) use a better way to check the type before casting
      assert(field->ob_type != NULL);
      assert(field->ob_type->tp_name != NULL);
      assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
      PyArrayObject * seq_field = (PyArrayObject *)field;
      Py_INCREF(seq_field);
      assert(PyArray_NDIM(seq_field) == 1);
      assert(PyArray_TYPE(seq_field) == NPY_FLOAT64);
      Py_ssize_t size = 6;
      double * dest = ros_message->f_min;
      for (Py_ssize_t i = 0; i < size; ++i) {
        double tmp = *(npy_float64 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(double));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // tau_max
    PyObject * field = PyObject_GetAttrString(_pymsg, "tau_max");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'tau_max'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!multi_mode_control_msgs__msg__JointArray__Sequence__init(&(ros_message->tau_max), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create multi_mode_control_msgs__msg__JointArray__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    multi_mode_control_msgs__msg__JointArray * dest = ros_message->tau_max.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!multi_mode_control_msgs__msg__joint_array__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }
  {  // tau_min
    PyObject * field = PyObject_GetAttrString(_pymsg, "tau_min");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'tau_min'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!multi_mode_control_msgs__msg__JointArray__Sequence__init(&(ros_message->tau_min), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create multi_mode_control_msgs__msg__JointArray__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    multi_mode_control_msgs__msg__JointArray * dest = ros_message->tau_min.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!multi_mode_control_msgs__msg__joint_array__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }
  {  // f_abs_max
    PyObject * field = PyObject_GetAttrString(_pymsg, "f_abs_max");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->f_abs_max = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // m_abs_max
    PyObject * field = PyObject_GetAttrString(_pymsg, "m_abs_max");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->m_abs_max = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // abs_mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "abs_mode");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->abs_mode = (Py_True == field);
    Py_DECREF(field);
  }
  {  // directional_mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "directional_mode");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->directional_mode = (Py_True == field);
    Py_DECREF(field);
  }
  {  // torque_mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "torque_mode");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->torque_mode = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * multi_mode_control_msgs__msg__contact_thresholds__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of ContactThresholds */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("multi_mode_control_msgs.msg._contact_thresholds");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "ContactThresholds");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  multi_mode_control_msgs__msg__ContactThresholds * ros_message = (multi_mode_control_msgs__msg__ContactThresholds *)raw_ros_message;
  {  // f_max
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "f_max");
    if (!field) {
      return NULL;
    }
    assert(field->ob_type != NULL);
    assert(field->ob_type->tp_name != NULL);
    assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
    PyArrayObject * seq_field = (PyArrayObject *)field;
    assert(PyArray_NDIM(seq_field) == 1);
    assert(PyArray_TYPE(seq_field) == NPY_FLOAT64);
    assert(sizeof(npy_float64) == sizeof(double));
    npy_float64 * dst = (npy_float64 *)PyArray_GETPTR1(seq_field, 0);
    double * src = &(ros_message->f_max[0]);
    memcpy(dst, src, 6 * sizeof(double));
    Py_DECREF(field);
  }
  {  // f_min
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "f_min");
    if (!field) {
      return NULL;
    }
    assert(field->ob_type != NULL);
    assert(field->ob_type->tp_name != NULL);
    assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
    PyArrayObject * seq_field = (PyArrayObject *)field;
    assert(PyArray_NDIM(seq_field) == 1);
    assert(PyArray_TYPE(seq_field) == NPY_FLOAT64);
    assert(sizeof(npy_float64) == sizeof(double));
    npy_float64 * dst = (npy_float64 *)PyArray_GETPTR1(seq_field, 0);
    double * src = &(ros_message->f_min[0]);
    memcpy(dst, src, 6 * sizeof(double));
    Py_DECREF(field);
  }
  {  // tau_max
    PyObject * field = NULL;
    size_t size = ros_message->tau_max.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    multi_mode_control_msgs__msg__JointArray * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->tau_max.data[i]);
      PyObject * pyitem = multi_mode_control_msgs__msg__joint_array__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "tau_max", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tau_min
    PyObject * field = NULL;
    size_t size = ros_message->tau_min.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    multi_mode_control_msgs__msg__JointArray * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->tau_min.data[i]);
      PyObject * pyitem = multi_mode_control_msgs__msg__joint_array__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "tau_min", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // f_abs_max
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->f_abs_max);
    {
      int rc = PyObject_SetAttrString(_pymessage, "f_abs_max", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // m_abs_max
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->m_abs_max);
    {
      int rc = PyObject_SetAttrString(_pymessage, "m_abs_max", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // abs_mode
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->abs_mode ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "abs_mode", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // directional_mode
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->directional_mode ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "directional_mode", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // torque_mode
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->torque_mode ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "torque_mode", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
