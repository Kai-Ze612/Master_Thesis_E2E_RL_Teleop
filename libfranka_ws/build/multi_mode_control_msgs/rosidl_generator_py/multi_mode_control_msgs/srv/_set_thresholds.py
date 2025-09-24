# generated from rosidl_generator_py/resource/_idl.py.em
# with input from multi_mode_control_msgs:srv/SetThresholds.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SetThresholds_Request(type):
    """Metaclass of message 'SetThresholds_Request'."""

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
                'multi_mode_control_msgs.srv.SetThresholds_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_thresholds__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_thresholds__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_thresholds__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_thresholds__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_thresholds__request

            from multi_mode_control_msgs.msg import ContactThresholds
            if ContactThresholds.__class__._TYPE_SUPPORT is None:
                ContactThresholds.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetThresholds_Request(metaclass=Metaclass_SetThresholds_Request):
    """Message class 'SetThresholds_Request'."""

    __slots__ = [
        '_level1',
        '_level2',
    ]

    _fields_and_field_types = {
        'level1': 'multi_mode_control_msgs/ContactThresholds',
        'level2': 'multi_mode_control_msgs/ContactThresholds',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['multi_mode_control_msgs', 'msg'], 'ContactThresholds'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['multi_mode_control_msgs', 'msg'], 'ContactThresholds'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from multi_mode_control_msgs.msg import ContactThresholds
        self.level1 = kwargs.get('level1', ContactThresholds())
        from multi_mode_control_msgs.msg import ContactThresholds
        self.level2 = kwargs.get('level2', ContactThresholds())

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
        if self.level1 != other.level1:
            return False
        if self.level2 != other.level2:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def level1(self):
        """Message field 'level1'."""
        return self._level1

    @level1.setter
    def level1(self, value):
        if __debug__:
            from multi_mode_control_msgs.msg import ContactThresholds
            assert \
                isinstance(value, ContactThresholds), \
                "The 'level1' field must be a sub message of type 'ContactThresholds'"
        self._level1 = value

    @builtins.property
    def level2(self):
        """Message field 'level2'."""
        return self._level2

    @level2.setter
    def level2(self, value):
        if __debug__:
            from multi_mode_control_msgs.msg import ContactThresholds
            assert \
                isinstance(value, ContactThresholds), \
                "The 'level2' field must be a sub message of type 'ContactThresholds'"
        self._level2 = value


# Import statements for member types

# already imported above
# import rosidl_parser.definition


class Metaclass_SetThresholds_Response(type):
    """Metaclass of message 'SetThresholds_Response'."""

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
                'multi_mode_control_msgs.srv.SetThresholds_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_thresholds__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_thresholds__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_thresholds__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_thresholds__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_thresholds__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetThresholds_Response(metaclass=Metaclass_SetThresholds_Response):
    """Message class 'SetThresholds_Response'."""

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


class Metaclass_SetThresholds(type):
    """Metaclass of service 'SetThresholds'."""

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
                'multi_mode_control_msgs.srv.SetThresholds')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__set_thresholds

            from multi_mode_control_msgs.srv import _set_thresholds
            if _set_thresholds.Metaclass_SetThresholds_Request._TYPE_SUPPORT is None:
                _set_thresholds.Metaclass_SetThresholds_Request.__import_type_support__()
            if _set_thresholds.Metaclass_SetThresholds_Response._TYPE_SUPPORT is None:
                _set_thresholds.Metaclass_SetThresholds_Response.__import_type_support__()


class SetThresholds(metaclass=Metaclass_SetThresholds):
    from multi_mode_control_msgs.srv._set_thresholds import SetThresholds_Request as Request
    from multi_mode_control_msgs.srv._set_thresholds import SetThresholds_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
