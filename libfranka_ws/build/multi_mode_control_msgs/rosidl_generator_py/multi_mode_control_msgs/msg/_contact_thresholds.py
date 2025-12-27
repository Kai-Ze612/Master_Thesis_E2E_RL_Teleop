# generated from rosidl_generator_py/resource/_idl.py.em
# with input from multi_mode_control_msgs:msg/ContactThresholds.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'f_max'
# Member 'f_min'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ContactThresholds(type):
    """Metaclass of message 'ContactThresholds'."""

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
            module = import_type_support('multi_mode_control_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'multi_mode_control_msgs.msg.ContactThresholds')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__contact_thresholds
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__contact_thresholds
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__contact_thresholds
            cls._TYPE_SUPPORT = module.type_support_msg__msg__contact_thresholds
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__contact_thresholds

            from multi_mode_control_msgs.msg import JointArray
            if JointArray.__class__._TYPE_SUPPORT is None:
                JointArray.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ContactThresholds(metaclass=Metaclass_ContactThresholds):
    """Message class 'ContactThresholds'."""

    __slots__ = [
        '_f_max',
        '_f_min',
        '_tau_max',
        '_tau_min',
        '_f_abs_max',
        '_m_abs_max',
        '_abs_mode',
        '_directional_mode',
        '_torque_mode',
    ]

    _fields_and_field_types = {
        'f_max': 'double[6]',
        'f_min': 'double[6]',
        'tau_max': 'sequence<multi_mode_control_msgs/JointArray>',
        'tau_min': 'sequence<multi_mode_control_msgs/JointArray>',
        'f_abs_max': 'double',
        'm_abs_max': 'double',
        'abs_mode': 'boolean',
        'directional_mode': 'boolean',
        'torque_mode': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['multi_mode_control_msgs', 'msg'], 'JointArray')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['multi_mode_control_msgs', 'msg'], 'JointArray')),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        if 'f_max' not in kwargs:
            self.f_max = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_max = numpy.array(kwargs.get('f_max'), dtype=numpy.float64)
            assert self.f_max.shape == (6, )
        if 'f_min' not in kwargs:
            self.f_min = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_min = numpy.array(kwargs.get('f_min'), dtype=numpy.float64)
            assert self.f_min.shape == (6, )
        self.tau_max = kwargs.get('tau_max', [])
        self.tau_min = kwargs.get('tau_min', [])
        self.f_abs_max = kwargs.get('f_abs_max', float())
        self.m_abs_max = kwargs.get('m_abs_max', float())
        self.abs_mode = kwargs.get('abs_mode', bool())
        self.directional_mode = kwargs.get('directional_mode', bool())
        self.torque_mode = kwargs.get('torque_mode', bool())

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
        if any(self.f_max != other.f_max):
            return False
        if any(self.f_min != other.f_min):
            return False
        if self.tau_max != other.tau_max:
            return False
        if self.tau_min != other.tau_min:
            return False
        if self.f_abs_max != other.f_abs_max:
            return False
        if self.m_abs_max != other.m_abs_max:
            return False
        if self.abs_mode != other.abs_mode:
            return False
        if self.directional_mode != other.directional_mode:
            return False
        if self.torque_mode != other.torque_mode:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def f_max(self):
        """Message field 'f_max'."""
        return self._f_max

    @f_max.setter
    def f_max(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_max' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_max' numpy.ndarray() must have a size of 6"
            self._f_max = value
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
                 len(value) == 6 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'f_max' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_max = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_min(self):
        """Message field 'f_min'."""
        return self._f_min

    @f_min.setter
    def f_min(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_min' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_min' numpy.ndarray() must have a size of 6"
            self._f_min = value
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
                 len(value) == 6 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'f_min' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_min = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def tau_max(self):
        """Message field 'tau_max'."""
        return self._tau_max

    @tau_max.setter
    def tau_max(self, value):
        if __debug__:
            from multi_mode_control_msgs.msg import JointArray
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
                 all(isinstance(v, JointArray) for v in value) and
                 True), \
                "The 'tau_max' field must be a set or sequence and each value of type 'JointArray'"
        self._tau_max = value

    @builtins.property
    def tau_min(self):
        """Message field 'tau_min'."""
        return self._tau_min

    @tau_min.setter
    def tau_min(self, value):
        if __debug__:
            from multi_mode_control_msgs.msg import JointArray
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
                 all(isinstance(v, JointArray) for v in value) and
                 True), \
                "The 'tau_min' field must be a set or sequence and each value of type 'JointArray'"
        self._tau_min = value

    @builtins.property
    def f_abs_max(self):
        """Message field 'f_abs_max'."""
        return self._f_abs_max

    @f_abs_max.setter
    def f_abs_max(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'f_abs_max' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'f_abs_max' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._f_abs_max = value

    @builtins.property
    def m_abs_max(self):
        """Message field 'm_abs_max'."""
        return self._m_abs_max

    @m_abs_max.setter
    def m_abs_max(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'm_abs_max' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'm_abs_max' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._m_abs_max = value

    @builtins.property
    def abs_mode(self):
        """Message field 'abs_mode'."""
        return self._abs_mode

    @abs_mode.setter
    def abs_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'abs_mode' field must be of type 'bool'"
        self._abs_mode = value

    @builtins.property
    def directional_mode(self):
        """Message field 'directional_mode'."""
        return self._directional_mode

    @directional_mode.setter
    def directional_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'directional_mode' field must be of type 'bool'"
        self._directional_mode = value

    @builtins.property
    def torque_mode(self):
        """Message field 'torque_mode'."""
        return self._torque_mode

    @torque_mode.setter
    def torque_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'torque_mode' field must be of type 'bool'"
        self._torque_mode = value
