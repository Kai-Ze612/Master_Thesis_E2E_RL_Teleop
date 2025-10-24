# generated from rosidl_generator_py/resource/_idl.py.em
# with input from mujoco_ros_msgs:msg/SimInfo.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SimInfo(type):
    """Metaclass of message 'SimInfo'."""

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
                'mujoco_ros_msgs.msg.SimInfo')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__sim_info
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__sim_info
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__sim_info
            cls._TYPE_SUPPORT = module.type_support_msg__msg__sim_info
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__sim_info

            from mujoco_ros_msgs.msg import StateUint
            if StateUint.__class__._TYPE_SUPPORT is None:
                StateUint.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SimInfo(metaclass=Metaclass_SimInfo):
    """Message class 'SimInfo'."""

    __slots__ = [
        '_model_path',
        '_model_valid',
        '_load_count',
        '_loading_state',
        '_paused',
        '_pending_sim_steps',
        '_rt_measured',
        '_rt_setting',
    ]

    _fields_and_field_types = {
        'model_path': 'string',
        'model_valid': 'boolean',
        'load_count': 'uint16',
        'loading_state': 'mujoco_ros_msgs/StateUint',
        'paused': 'boolean',
        'pending_sim_steps': 'uint16',
        'rt_measured': 'float',
        'rt_setting': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['mujoco_ros_msgs', 'msg'], 'StateUint'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.model_path = kwargs.get('model_path', str())
        self.model_valid = kwargs.get('model_valid', bool())
        self.load_count = kwargs.get('load_count', int())
        from mujoco_ros_msgs.msg import StateUint
        self.loading_state = kwargs.get('loading_state', StateUint())
        self.paused = kwargs.get('paused', bool())
        self.pending_sim_steps = kwargs.get('pending_sim_steps', int())
        self.rt_measured = kwargs.get('rt_measured', float())
        self.rt_setting = kwargs.get('rt_setting', float())

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
        if self.model_path != other.model_path:
            return False
        if self.model_valid != other.model_valid:
            return False
        if self.load_count != other.load_count:
            return False
        if self.loading_state != other.loading_state:
            return False
        if self.paused != other.paused:
            return False
        if self.pending_sim_steps != other.pending_sim_steps:
            return False
        if self.rt_measured != other.rt_measured:
            return False
        if self.rt_setting != other.rt_setting:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def model_path(self):
        """Message field 'model_path'."""
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'model_path' field must be of type 'str'"
        self._model_path = value

    @builtins.property
    def model_valid(self):
        """Message field 'model_valid'."""
        return self._model_valid

    @model_valid.setter
    def model_valid(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'model_valid' field must be of type 'bool'"
        self._model_valid = value

    @builtins.property
    def load_count(self):
        """Message field 'load_count'."""
        return self._load_count

    @load_count.setter
    def load_count(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'load_count' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'load_count' field must be an unsigned integer in [0, 65535]"
        self._load_count = value

    @builtins.property
    def loading_state(self):
        """Message field 'loading_state'."""
        return self._loading_state

    @loading_state.setter
    def loading_state(self, value):
        if __debug__:
            from mujoco_ros_msgs.msg import StateUint
            assert \
                isinstance(value, StateUint), \
                "The 'loading_state' field must be a sub message of type 'StateUint'"
        self._loading_state = value

    @builtins.property
    def paused(self):
        """Message field 'paused'."""
        return self._paused

    @paused.setter
    def paused(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'paused' field must be of type 'bool'"
        self._paused = value

    @builtins.property
    def pending_sim_steps(self):
        """Message field 'pending_sim_steps'."""
        return self._pending_sim_steps

    @pending_sim_steps.setter
    def pending_sim_steps(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'pending_sim_steps' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'pending_sim_steps' field must be an unsigned integer in [0, 65535]"
        self._pending_sim_steps = value

    @builtins.property
    def rt_measured(self):
        """Message field 'rt_measured'."""
        return self._rt_measured

    @rt_measured.setter
    def rt_measured(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rt_measured' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'rt_measured' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._rt_measured = value

    @builtins.property
    def rt_setting(self):
        """Message field 'rt_setting'."""
        return self._rt_setting

    @rt_setting.setter
    def rt_setting(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rt_setting' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'rt_setting' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._rt_setting = value
