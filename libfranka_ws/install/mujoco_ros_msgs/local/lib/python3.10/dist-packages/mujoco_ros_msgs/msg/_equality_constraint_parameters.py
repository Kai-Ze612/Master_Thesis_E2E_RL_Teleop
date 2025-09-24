# generated from rosidl_generator_py/resource/_idl.py.em
# with input from mujoco_ros_msgs:msg/EqualityConstraintParameters.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'polycoef'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_EqualityConstraintParameters(type):
    """Metaclass of message 'EqualityConstraintParameters'."""

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
                'mujoco_ros_msgs.msg.EqualityConstraintParameters')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__equality_constraint_parameters
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__equality_constraint_parameters
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__equality_constraint_parameters
            cls._TYPE_SUPPORT = module.type_support_msg__msg__equality_constraint_parameters
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__equality_constraint_parameters

            from geometry_msgs.msg import Pose
            if Pose.__class__._TYPE_SUPPORT is None:
                Pose.__class__.__import_type_support__()

            from geometry_msgs.msg import Vector3
            if Vector3.__class__._TYPE_SUPPORT is None:
                Vector3.__class__.__import_type_support__()

            from mujoco_ros_msgs.msg import EqualityConstraintType
            if EqualityConstraintType.__class__._TYPE_SUPPORT is None:
                EqualityConstraintType.__class__.__import_type_support__()

            from mujoco_ros_msgs.msg import SolverParameters
            if SolverParameters.__class__._TYPE_SUPPORT is None:
                SolverParameters.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class EqualityConstraintParameters(metaclass=Metaclass_EqualityConstraintParameters):
    """Message class 'EqualityConstraintParameters'."""

    __slots__ = [
        '_name',
        '_type',
        '_solver_parameters',
        '_active',
        '_class_param',
        '_element1',
        '_element2',
        '_torquescale',
        '_anchor',
        '_relpose',
        '_polycoef',
    ]

    _fields_and_field_types = {
        'name': 'string',
        'type': 'mujoco_ros_msgs/EqualityConstraintType',
        'solver_parameters': 'mujoco_ros_msgs/SolverParameters',
        'active': 'boolean',
        'class_param': 'string',
        'element1': 'string',
        'element2': 'string',
        'torquescale': 'double',
        'anchor': 'geometry_msgs/Vector3',
        'relpose': 'geometry_msgs/Pose',
        'polycoef': 'sequence<double>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['mujoco_ros_msgs', 'msg'], 'EqualityConstraintType'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['mujoco_ros_msgs', 'msg'], 'SolverParameters'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Pose'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.name = kwargs.get('name', str())
        from mujoco_ros_msgs.msg import EqualityConstraintType
        self.type = kwargs.get('type', EqualityConstraintType())
        from mujoco_ros_msgs.msg import SolverParameters
        self.solver_parameters = kwargs.get('solver_parameters', SolverParameters())
        self.active = kwargs.get('active', bool())
        self.class_param = kwargs.get('class_param', str())
        self.element1 = kwargs.get('element1', str())
        self.element2 = kwargs.get('element2', str())
        self.torquescale = kwargs.get('torquescale', float())
        from geometry_msgs.msg import Vector3
        self.anchor = kwargs.get('anchor', Vector3())
        from geometry_msgs.msg import Pose
        self.relpose = kwargs.get('relpose', Pose())
        self.polycoef = array.array('d', kwargs.get('polycoef', []))

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
        if self.name != other.name:
            return False
        if self.type != other.type:
            return False
        if self.solver_parameters != other.solver_parameters:
            return False
        if self.active != other.active:
            return False
        if self.class_param != other.class_param:
            return False
        if self.element1 != other.element1:
            return False
        if self.element2 != other.element2:
            return False
        if self.torquescale != other.torquescale:
            return False
        if self.anchor != other.anchor:
            return False
        if self.relpose != other.relpose:
            return False
        if self.polycoef != other.polycoef:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def name(self):
        """Message field 'name'."""
        return self._name

    @name.setter
    def name(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'name' field must be of type 'str'"
        self._name = value

    @builtins.property  # noqa: A003
    def type(self):  # noqa: A003
        """Message field 'type'."""
        return self._type

    @type.setter  # noqa: A003
    def type(self, value):  # noqa: A003
        if __debug__:
            from mujoco_ros_msgs.msg import EqualityConstraintType
            assert \
                isinstance(value, EqualityConstraintType), \
                "The 'type' field must be a sub message of type 'EqualityConstraintType'"
        self._type = value

    @builtins.property
    def solver_parameters(self):
        """Message field 'solver_parameters'."""
        return self._solver_parameters

    @solver_parameters.setter
    def solver_parameters(self, value):
        if __debug__:
            from mujoco_ros_msgs.msg import SolverParameters
            assert \
                isinstance(value, SolverParameters), \
                "The 'solver_parameters' field must be a sub message of type 'SolverParameters'"
        self._solver_parameters = value

    @builtins.property
    def active(self):
        """Message field 'active'."""
        return self._active

    @active.setter
    def active(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'active' field must be of type 'bool'"
        self._active = value

    @builtins.property
    def class_param(self):
        """Message field 'class_param'."""
        return self._class_param

    @class_param.setter
    def class_param(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'class_param' field must be of type 'str'"
        self._class_param = value

    @builtins.property
    def element1(self):
        """Message field 'element1'."""
        return self._element1

    @element1.setter
    def element1(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'element1' field must be of type 'str'"
        self._element1 = value

    @builtins.property
    def element2(self):
        """Message field 'element2'."""
        return self._element2

    @element2.setter
    def element2(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'element2' field must be of type 'str'"
        self._element2 = value

    @builtins.property
    def torquescale(self):
        """Message field 'torquescale'."""
        return self._torquescale

    @torquescale.setter
    def torquescale(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'torquescale' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'torquescale' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._torquescale = value

    @builtins.property
    def anchor(self):
        """Message field 'anchor'."""
        return self._anchor

    @anchor.setter
    def anchor(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'anchor' field must be a sub message of type 'Vector3'"
        self._anchor = value

    @builtins.property
    def relpose(self):
        """Message field 'relpose'."""
        return self._relpose

    @relpose.setter
    def relpose(self, value):
        if __debug__:
            from geometry_msgs.msg import Pose
            assert \
                isinstance(value, Pose), \
                "The 'relpose' field must be a sub message of type 'Pose'"
        self._relpose = value

    @builtins.property
    def polycoef(self):
        """Message field 'polycoef'."""
        return self._polycoef

    @polycoef.setter
    def polycoef(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'polycoef' array.array() must have the type code of 'd'"
            self._polycoef = value
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'polycoef' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._polycoef = array.array('d', value)
