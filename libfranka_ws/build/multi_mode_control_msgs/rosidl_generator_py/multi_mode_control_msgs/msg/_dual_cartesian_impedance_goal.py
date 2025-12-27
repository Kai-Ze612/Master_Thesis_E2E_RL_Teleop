# generated from rosidl_generator_py/resource/_idl.py.em
# with input from multi_mode_control_msgs:msg/DualCartesianImpedanceGoal.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'l_q_n'
# Member 'r_q_n'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_DualCartesianImpedanceGoal(type):
    """Metaclass of message 'DualCartesianImpedanceGoal'."""

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
                'multi_mode_control_msgs.msg.DualCartesianImpedanceGoal')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__dual_cartesian_impedance_goal
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__dual_cartesian_impedance_goal
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__dual_cartesian_impedance_goal
            cls._TYPE_SUPPORT = module.type_support_msg__msg__dual_cartesian_impedance_goal
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__dual_cartesian_impedance_goal

            from geometry_msgs.msg import Pose
            if Pose.__class__._TYPE_SUPPORT is None:
                Pose.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class DualCartesianImpedanceGoal(metaclass=Metaclass_DualCartesianImpedanceGoal):
    """Message class 'DualCartesianImpedanceGoal'."""

    __slots__ = [
        '_l_pose',
        '_r_pose',
        '_l_q_n',
        '_r_q_n',
    ]

    _fields_and_field_types = {
        'l_pose': 'geometry_msgs/Pose',
        'r_pose': 'geometry_msgs/Pose',
        'l_q_n': 'double[7]',
        'r_q_n': 'double[7]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Pose'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Pose'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 7),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 7),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from geometry_msgs.msg import Pose
        self.l_pose = kwargs.get('l_pose', Pose())
        from geometry_msgs.msg import Pose
        self.r_pose = kwargs.get('r_pose', Pose())
        if 'l_q_n' not in kwargs:
            self.l_q_n = numpy.zeros(7, dtype=numpy.float64)
        else:
            self.l_q_n = numpy.array(kwargs.get('l_q_n'), dtype=numpy.float64)
            assert self.l_q_n.shape == (7, )
        if 'r_q_n' not in kwargs:
            self.r_q_n = numpy.zeros(7, dtype=numpy.float64)
        else:
            self.r_q_n = numpy.array(kwargs.get('r_q_n'), dtype=numpy.float64)
            assert self.r_q_n.shape == (7, )

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
        if self.l_pose != other.l_pose:
            return False
        if self.r_pose != other.r_pose:
            return False
        if any(self.l_q_n != other.l_q_n):
            return False
        if any(self.r_q_n != other.r_q_n):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def l_pose(self):
        """Message field 'l_pose'."""
        return self._l_pose

    @l_pose.setter
    def l_pose(self, value):
        if __debug__:
            from geometry_msgs.msg import Pose
            assert \
                isinstance(value, Pose), \
                "The 'l_pose' field must be a sub message of type 'Pose'"
        self._l_pose = value

    @builtins.property
    def r_pose(self):
        """Message field 'r_pose'."""
        return self._r_pose

    @r_pose.setter
    def r_pose(self, value):
        if __debug__:
            from geometry_msgs.msg import Pose
            assert \
                isinstance(value, Pose), \
                "The 'r_pose' field must be a sub message of type 'Pose'"
        self._r_pose = value

    @builtins.property
    def l_q_n(self):
        """Message field 'l_q_n'."""
        return self._l_q_n

    @l_q_n.setter
    def l_q_n(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'l_q_n' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 7, \
                "The 'l_q_n' numpy.ndarray() must have a size of 7"
            self._l_q_n = value
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
                "The 'l_q_n' field must be a set or sequence with length 7 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._l_q_n = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def r_q_n(self):
        """Message field 'r_q_n'."""
        return self._r_q_n

    @r_q_n.setter
    def r_q_n(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'r_q_n' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 7, \
                "The 'r_q_n' numpy.ndarray() must have a size of 7"
            self._r_q_n = value
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
                "The 'r_q_n' field must be a set or sequence with length 7 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._r_q_n = numpy.array(value, dtype=numpy.float64)
