# generated from rosidl_generator_py/resource/_idl.py.em
# with input from mujoco_ros_msgs:msg/GeomProperties.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_GeomProperties(type):
    """Metaclass of message 'GeomProperties'."""

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
                'mujoco_ros_msgs.msg.GeomProperties')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__geom_properties
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__geom_properties
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__geom_properties
            cls._TYPE_SUPPORT = module.type_support_msg__msg__geom_properties
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__geom_properties

            from mujoco_ros_msgs.msg import GeomType
            if GeomType.__class__._TYPE_SUPPORT is None:
                GeomType.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class GeomProperties(metaclass=Metaclass_GeomProperties):
    """Message class 'GeomProperties'."""

    __slots__ = [
        '_name',
        '_type',
        '_body_mass',
        '_size_0',
        '_size_1',
        '_size_2',
        '_friction_slide',
        '_friction_spin',
        '_friction_roll',
    ]

    _fields_and_field_types = {
        'name': 'string',
        'type': 'mujoco_ros_msgs/GeomType',
        'body_mass': 'float',
        'size_0': 'float',
        'size_1': 'float',
        'size_2': 'float',
        'friction_slide': 'float',
        'friction_spin': 'float',
        'friction_roll': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['mujoco_ros_msgs', 'msg'], 'GeomType'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.name = kwargs.get('name', str())
        from mujoco_ros_msgs.msg import GeomType
        self.type = kwargs.get('type', GeomType())
        self.body_mass = kwargs.get('body_mass', float())
        self.size_0 = kwargs.get('size_0', float())
        self.size_1 = kwargs.get('size_1', float())
        self.size_2 = kwargs.get('size_2', float())
        self.friction_slide = kwargs.get('friction_slide', float())
        self.friction_spin = kwargs.get('friction_spin', float())
        self.friction_roll = kwargs.get('friction_roll', float())

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
        if self.body_mass != other.body_mass:
            return False
        if self.size_0 != other.size_0:
            return False
        if self.size_1 != other.size_1:
            return False
        if self.size_2 != other.size_2:
            return False
        if self.friction_slide != other.friction_slide:
            return False
        if self.friction_spin != other.friction_spin:
            return False
        if self.friction_roll != other.friction_roll:
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
            from mujoco_ros_msgs.msg import GeomType
            assert \
                isinstance(value, GeomType), \
                "The 'type' field must be a sub message of type 'GeomType'"
        self._type = value

    @builtins.property
    def body_mass(self):
        """Message field 'body_mass'."""
        return self._body_mass

    @body_mass.setter
    def body_mass(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'body_mass' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'body_mass' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._body_mass = value

    @builtins.property
    def size_0(self):
        """Message field 'size_0'."""
        return self._size_0

    @size_0.setter
    def size_0(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'size_0' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'size_0' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._size_0 = value

    @builtins.property
    def size_1(self):
        """Message field 'size_1'."""
        return self._size_1

    @size_1.setter
    def size_1(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'size_1' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'size_1' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._size_1 = value

    @builtins.property
    def size_2(self):
        """Message field 'size_2'."""
        return self._size_2

    @size_2.setter
    def size_2(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'size_2' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'size_2' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._size_2 = value

    @builtins.property
    def friction_slide(self):
        """Message field 'friction_slide'."""
        return self._friction_slide

    @friction_slide.setter
    def friction_slide(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'friction_slide' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'friction_slide' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._friction_slide = value

    @builtins.property
    def friction_spin(self):
        """Message field 'friction_spin'."""
        return self._friction_spin

    @friction_spin.setter
    def friction_spin(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'friction_spin' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'friction_spin' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._friction_spin = value

    @builtins.property
    def friction_roll(self):
        """Message field 'friction_roll'."""
        return self._friction_roll

    @friction_roll.setter
    def friction_roll(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'friction_roll' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'friction_roll' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._friction_roll = value
