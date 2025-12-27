# generated from rosidl_generator_py/resource/_idl.py.em
# with input from mujoco_ros_msgs:msg/PluginStats.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PluginStats(type):
    """Metaclass of message 'PluginStats'."""

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
                'mujoco_ros_msgs.msg.PluginStats')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__plugin_stats
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__plugin_stats
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__plugin_stats
            cls._TYPE_SUPPORT = module.type_support_msg__msg__plugin_stats
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__plugin_stats

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PluginStats(metaclass=Metaclass_PluginStats):
    """Message class 'PluginStats'."""

    __slots__ = [
        '_plugin_type',
        '_load_time',
        '_reset_time',
        '_ema_steptime_control',
        '_ema_steptime_passive',
        '_ema_steptime_render',
        '_ema_steptime_last_stage',
    ]

    _fields_and_field_types = {
        'plugin_type': 'string',
        'load_time': 'float',
        'reset_time': 'float',
        'ema_steptime_control': 'float',
        'ema_steptime_passive': 'float',
        'ema_steptime_render': 'float',
        'ema_steptime_last_stage': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
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
        self.plugin_type = kwargs.get('plugin_type', str())
        self.load_time = kwargs.get('load_time', float())
        self.reset_time = kwargs.get('reset_time', float())
        self.ema_steptime_control = kwargs.get('ema_steptime_control', float())
        self.ema_steptime_passive = kwargs.get('ema_steptime_passive', float())
        self.ema_steptime_render = kwargs.get('ema_steptime_render', float())
        self.ema_steptime_last_stage = kwargs.get('ema_steptime_last_stage', float())

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
        if self.plugin_type != other.plugin_type:
            return False
        if self.load_time != other.load_time:
            return False
        if self.reset_time != other.reset_time:
            return False
        if self.ema_steptime_control != other.ema_steptime_control:
            return False
        if self.ema_steptime_passive != other.ema_steptime_passive:
            return False
        if self.ema_steptime_render != other.ema_steptime_render:
            return False
        if self.ema_steptime_last_stage != other.ema_steptime_last_stage:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def plugin_type(self):
        """Message field 'plugin_type'."""
        return self._plugin_type

    @plugin_type.setter
    def plugin_type(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'plugin_type' field must be of type 'str'"
        self._plugin_type = value

    @builtins.property
    def load_time(self):
        """Message field 'load_time'."""
        return self._load_time

    @load_time.setter
    def load_time(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'load_time' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'load_time' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._load_time = value

    @builtins.property
    def reset_time(self):
        """Message field 'reset_time'."""
        return self._reset_time

    @reset_time.setter
    def reset_time(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'reset_time' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'reset_time' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._reset_time = value

    @builtins.property
    def ema_steptime_control(self):
        """Message field 'ema_steptime_control'."""
        return self._ema_steptime_control

    @ema_steptime_control.setter
    def ema_steptime_control(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ema_steptime_control' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'ema_steptime_control' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._ema_steptime_control = value

    @builtins.property
    def ema_steptime_passive(self):
        """Message field 'ema_steptime_passive'."""
        return self._ema_steptime_passive

    @ema_steptime_passive.setter
    def ema_steptime_passive(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ema_steptime_passive' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'ema_steptime_passive' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._ema_steptime_passive = value

    @builtins.property
    def ema_steptime_render(self):
        """Message field 'ema_steptime_render'."""
        return self._ema_steptime_render

    @ema_steptime_render.setter
    def ema_steptime_render(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ema_steptime_render' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'ema_steptime_render' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._ema_steptime_render = value

    @builtins.property
    def ema_steptime_last_stage(self):
        """Message field 'ema_steptime_last_stage'."""
        return self._ema_steptime_last_stage

    @ema_steptime_last_stage.setter
    def ema_steptime_last_stage(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ema_steptime_last_stage' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'ema_steptime_last_stage' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._ema_steptime_last_stage = value
