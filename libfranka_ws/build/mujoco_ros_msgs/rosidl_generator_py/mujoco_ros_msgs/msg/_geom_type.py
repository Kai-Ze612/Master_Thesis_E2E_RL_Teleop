# generated from rosidl_generator_py/resource/_idl.py.em
# with input from mujoco_ros_msgs:msg/GeomType.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_GeomType(type):
    """Metaclass of message 'GeomType'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'PLANE': 0,
        'HFIELD': 1,
        'SPHERE': 2,
        'CAPSULE': 3,
        'ELLIPSOID': 4,
        'CYLINDER': 5,
        'BOX': 6,
        'MESH': 7,
        'GEOM_NONE': 1001,
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
                'mujoco_ros_msgs.msg.GeomType')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__geom_type
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__geom_type
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__geom_type
            cls._TYPE_SUPPORT = module.type_support_msg__msg__geom_type
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__geom_type

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'PLANE': cls.__constants['PLANE'],
            'HFIELD': cls.__constants['HFIELD'],
            'SPHERE': cls.__constants['SPHERE'],
            'CAPSULE': cls.__constants['CAPSULE'],
            'ELLIPSOID': cls.__constants['ELLIPSOID'],
            'CYLINDER': cls.__constants['CYLINDER'],
            'BOX': cls.__constants['BOX'],
            'MESH': cls.__constants['MESH'],
            'GEOM_NONE': cls.__constants['GEOM_NONE'],
        }

    @property
    def PLANE(self):
        """Message constant 'PLANE'."""
        return Metaclass_GeomType.__constants['PLANE']

    @property
    def HFIELD(self):
        """Message constant 'HFIELD'."""
        return Metaclass_GeomType.__constants['HFIELD']

    @property
    def SPHERE(self):
        """Message constant 'SPHERE'."""
        return Metaclass_GeomType.__constants['SPHERE']

    @property
    def CAPSULE(self):
        """Message constant 'CAPSULE'."""
        return Metaclass_GeomType.__constants['CAPSULE']

    @property
    def ELLIPSOID(self):
        """Message constant 'ELLIPSOID'."""
        return Metaclass_GeomType.__constants['ELLIPSOID']

    @property
    def CYLINDER(self):
        """Message constant 'CYLINDER'."""
        return Metaclass_GeomType.__constants['CYLINDER']

    @property
    def BOX(self):
        """Message constant 'BOX'."""
        return Metaclass_GeomType.__constants['BOX']

    @property
    def MESH(self):
        """Message constant 'MESH'."""
        return Metaclass_GeomType.__constants['MESH']

    @property
    def GEOM_NONE(self):
        """Message constant 'GEOM_NONE'."""
        return Metaclass_GeomType.__constants['GEOM_NONE']


class GeomType(metaclass=Metaclass_GeomType):
    """
    Message class 'GeomType'.

    Constants:
      PLANE
      HFIELD
      SPHERE
      CAPSULE
      ELLIPSOID
      CYLINDER
      BOX
      MESH
      GEOM_NONE
    """

    __slots__ = [
        '_value',
    ]

    _fields_and_field_types = {
        'value': 'uint16',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.value = kwargs.get('value', int())

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
        if self.value != other.value:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def value(self):
        """Message field 'value'."""
        return self._value

    @value.setter
    def value(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'value' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'value' field must be an unsigned integer in [0, 65535]"
        self._value = value
