# generated from rosidl_generator_py/resource/_idl.py.em
# with input from multi_mode_control_msgs:srv/SetCartesianImpedance.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'stiffness'
# Member 'damping_ratio'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SetCartesianImpedance_Request(type):
    """Metaclass of message 'SetCartesianImpedance_Request'."""

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
                'multi_mode_control_msgs.srv.SetCartesianImpedance_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_cartesian_impedance__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_cartesian_impedance__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_cartesian_impedance__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_cartesian_impedance__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_cartesian_impedance__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetCartesianImpedance_Request(metaclass=Metaclass_SetCartesianImpedance_Request):
    """Message class 'SetCartesianImpedance_Request'."""

    __slots__ = [
        '_stiffness',
        '_damping_ratio',
        '_nullspace_stiffness',
    ]

    _fields_and_field_types = {
        'stiffness': 'double[36]',
        'damping_ratio': 'double[6]',
        'nullspace_stiffness': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 36),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        if 'stiffness' not in kwargs:
            self.stiffness = numpy.zeros(36, dtype=numpy.float64)
        else:
            self.stiffness = numpy.array(kwargs.get('stiffness'), dtype=numpy.float64)
            assert self.stiffness.shape == (36, )
        if 'damping_ratio' not in kwargs:
            self.damping_ratio = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.damping_ratio = numpy.array(kwargs.get('damping_ratio'), dtype=numpy.float64)
            assert self.damping_ratio.shape == (6, )
        self.nullspace_stiffness = kwargs.get('nullspace_stiffness', float())

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
        if any(self.stiffness != other.stiffness):
            return False
        if any(self.damping_ratio != other.damping_ratio):
            return False
        if self.nullspace_stiffness != other.nullspace_stiffness:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def stiffness(self):
        """Message field 'stiffness'."""
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'stiffness' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 36, \
                "The 'stiffness' numpy.ndarray() must have a size of 36"
            self._stiffness = value
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
                 len(value) == 36 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'stiffness' field must be a set or sequence with length 36 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._stiffness = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def damping_ratio(self):
        """Message field 'damping_ratio'."""
        return self._damping_ratio

    @damping_ratio.setter
    def damping_ratio(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'damping_ratio' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'damping_ratio' numpy.ndarray() must have a size of 6"
            self._damping_ratio = value
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
                "The 'damping_ratio' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._damping_ratio = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def nullspace_stiffness(self):
        """Message field 'nullspace_stiffness'."""
        return self._nullspace_stiffness

    @nullspace_stiffness.setter
    def nullspace_stiffness(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'nullspace_stiffness' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'nullspace_stiffness' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._nullspace_stiffness = value


# Import statements for member types

# already imported above
# import rosidl_parser.definition


class Metaclass_SetCartesianImpedance_Response(type):
    """Metaclass of message 'SetCartesianImpedance_Response'."""

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
                'multi_mode_control_msgs.srv.SetCartesianImpedance_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_cartesian_impedance__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_cartesian_impedance__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_cartesian_impedance__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_cartesian_impedance__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_cartesian_impedance__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetCartesianImpedance_Response(metaclass=Metaclass_SetCartesianImpedance_Response):
    """Message class 'SetCartesianImpedance_Response'."""

    __slots__ = [
    ]

    _fields_and_field_types = {
    }

    SLOT_TYPES = (
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))

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
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)


class Metaclass_SetCartesianImpedance(type):
    """Metaclass of service 'SetCartesianImpedance'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('multi_mode_control_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'multi_mode_control_msgs.srv.SetCartesianImpedance')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__set_cartesian_impedance

            from multi_mode_control_msgs.srv import _set_cartesian_impedance
            if _set_cartesian_impedance.Metaclass_SetCartesianImpedance_Request._TYPE_SUPPORT is None:
                _set_cartesian_impedance.Metaclass_SetCartesianImpedance_Request.__import_type_support__()
            if _set_cartesian_impedance.Metaclass_SetCartesianImpedance_Response._TYPE_SUPPORT is None:
                _set_cartesian_impedance.Metaclass_SetCartesianImpedance_Response.__import_type_support__()


class SetCartesianImpedance(metaclass=Metaclass_SetCartesianImpedance):
    from multi_mode_control_msgs.srv._set_cartesian_impedance import SetCartesianImpedance_Request as Request
    from multi_mode_control_msgs.srv._set_cartesian_impedance import SetCartesianImpedance_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
