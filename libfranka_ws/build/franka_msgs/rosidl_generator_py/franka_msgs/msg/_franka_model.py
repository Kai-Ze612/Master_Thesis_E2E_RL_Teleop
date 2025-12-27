# generated from rosidl_generator_py/resource/_idl.py.em
# with input from franka_msgs:msg/FrankaModel.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'coriolis'
# Member 'ee_zero_jacobian'
# Member 'ee_body_jacobian'
# Member 'mass'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_FrankaModel(type):
    """Metaclass of message 'FrankaModel'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('franka_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'franka_msgs.msg.FrankaModel')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__franka_model
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__franka_model
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__franka_model
            cls._TYPE_SUPPORT = module.type_support_msg__msg__franka_model
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__franka_model

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class FrankaModel(metaclass=Metaclass_FrankaModel):
    """Message class 'FrankaModel'."""

    __slots__ = [
        '_header',
        '_coriolis',
        '_ee_zero_jacobian',
        '_ee_body_jacobian',
        '_mass',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'coriolis': 'double[7]',
        'ee_zero_jacobian': 'double[42]',
        'ee_body_jacobian': 'double[42]',
        'mass': 'double[49]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 7),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 42),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 42),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 49),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        if 'coriolis' not in kwargs:
            self.coriolis = numpy.zeros(7, dtype=numpy.float64)
        else:
            self.coriolis = numpy.array(kwargs.get('coriolis'), dtype=numpy.float64)
            assert self.coriolis.shape == (7, )
        if 'ee_zero_jacobian' not in kwargs:
            self.ee_zero_jacobian = numpy.zeros(42, dtype=numpy.float64)
        else:
            self.ee_zero_jacobian = numpy.array(kwargs.get('ee_zero_jacobian'), dtype=numpy.float64)
            assert self.ee_zero_jacobian.shape == (42, )
        if 'ee_body_jacobian' not in kwargs:
            self.ee_body_jacobian = numpy.zeros(42, dtype=numpy.float64)
        else:
            self.ee_body_jacobian = numpy.array(kwargs.get('ee_body_jacobian'), dtype=numpy.float64)
            assert self.ee_body_jacobian.shape == (42, )
        if 'mass' not in kwargs:
            self.mass = numpy.zeros(49, dtype=numpy.float64)
        else:
            self.mass = numpy.array(kwargs.get('mass'), dtype=numpy.float64)
            assert self.mass.shape == (49, )

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if any(self.coriolis != other.coriolis):
            return False
        if any(self.ee_zero_jacobian != other.ee_zero_jacobian):
            return False
        if any(self.ee_body_jacobian != other.ee_body_jacobian):
            return False
        if any(self.mass != other.mass):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def coriolis(self):
        """Message field 'coriolis'."""
        return self._coriolis

    @coriolis.setter
    def coriolis(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'coriolis' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 7, \
                "The 'coriolis' numpy.ndarray() must have a size of 7"
            self._coriolis = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 7 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'coriolis' field must be a set or sequence with length 7 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._coriolis = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def ee_zero_jacobian(self):
        """Message field 'ee_zero_jacobian'."""
        return self._ee_zero_jacobian

    @ee_zero_jacobian.setter
    def ee_zero_jacobian(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'ee_zero_jacobian' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 42, \
                "The 'ee_zero_jacobian' numpy.ndarray() must have a size of 42"
            self._ee_zero_jacobian = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 42 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'ee_zero_jacobian' field must be a set or sequence with length 42 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._ee_zero_jacobian = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def ee_body_jacobian(self):
        """Message field 'ee_body_jacobian'."""
        return self._ee_body_jacobian

    @ee_body_jacobian.setter
    def ee_body_jacobian(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'ee_body_jacobian' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 42, \
                "The 'ee_body_jacobian' numpy.ndarray() must have a size of 42"
            self._ee_body_jacobian = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 42 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'ee_body_jacobian' field must be a set or sequence with length 42 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._ee_body_jacobian = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def mass(self):
        """Message field 'mass'."""
        return self._mass

    @mass.setter
    def mass(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'mass' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 49, \
                "The 'mass' numpy.ndarray() must have a size of 49"
            self._mass = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 49 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'mass' field must be a set or sequence with length 49 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._mass = numpy.array(value, dtype=numpy.float64)
