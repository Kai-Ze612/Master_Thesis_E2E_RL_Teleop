# generated from rosidl_generator_py/resource/_idl.py.em
# with input from mujoco_ros_msgs:msg/SolverParameters.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SolverParameters(type):
    """Metaclass of message 'SolverParameters'."""

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
            module = import_type_support('mujoco_ros_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'mujoco_ros_msgs.msg.SolverParameters')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__solver_parameters
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__solver_parameters
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__solver_parameters
            cls._TYPE_SUPPORT = module.type_support_msg__msg__solver_parameters
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__solver_parameters

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SolverParameters(metaclass=Metaclass_SolverParameters):
    """Message class 'SolverParameters'."""

    __slots__ = [
        '_dmin',
        '_dmax',
        '_width',
        '_midpoint',
        '_power',
        '_timeconst',
        '_dampratio',
    ]

    _fields_and_field_types = {
        'dmin': 'double',
        'dmax': 'double',
        'width': 'double',
        'midpoint': 'double',
        'power': 'double',
        'timeconst': 'double',
        'dampratio': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.dmin = kwargs.get('dmin', float())
        self.dmax = kwargs.get('dmax', float())
        self.width = kwargs.get('width', float())
        self.midpoint = kwargs.get('midpoint', float())
        self.power = kwargs.get('power', float())
        self.timeconst = kwargs.get('timeconst', float())
        self.dampratio = kwargs.get('dampratio', float())

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
        if self.dmin != other.dmin:
            return False
        if self.dmax != other.dmax:
            return False
        if self.width != other.width:
            return False
        if self.midpoint != other.midpoint:
            return False
        if self.power != other.power:
            return False
        if self.timeconst != other.timeconst:
            return False
        if self.dampratio != other.dampratio:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def dmin(self):
        """Message field 'dmin'."""
        return self._dmin

    @dmin.setter
    def dmin(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'dmin' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'dmin' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._dmin = value

    @builtins.property
    def dmax(self):
        """Message field 'dmax'."""
        return self._dmax

    @dmax.setter
    def dmax(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'dmax' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'dmax' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._dmax = value

    @builtins.property
    def width(self):
        """Message field 'width'."""
        return self._width

    @width.setter
    def width(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'width' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'width' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._width = value

    @builtins.property
    def midpoint(self):
        """Message field 'midpoint'."""
        return self._midpoint

    @midpoint.setter
    def midpoint(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'midpoint' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'midpoint' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._midpoint = value

    @builtins.property
    def power(self):
        """Message field 'power'."""
        return self._power

    @power.setter
    def power(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'power' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'power' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._power = value

    @builtins.property
    def timeconst(self):
        """Message field 'timeconst'."""
        return self._timeconst

    @timeconst.setter
    def timeconst(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'timeconst' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'timeconst' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._timeconst = value

    @builtins.property
    def dampratio(self):
        """Message field 'dampratio'."""
        return self._dampratio

    @dampratio.setter
    def dampratio(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'dampratio' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'dampratio' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._dampratio = value
