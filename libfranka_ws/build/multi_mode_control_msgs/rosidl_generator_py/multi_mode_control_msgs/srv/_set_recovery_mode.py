# generated from rosidl_generator_py/resource/_idl.py.em
# with input from multi_mode_control_msgs:srv/SetRecoveryMode.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SetRecoveryMode_Request(type):
    """Metaclass of message 'SetRecoveryMode_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'AUTO': 0,
        'MANUAL': 1,
        'PROGRAM': 2,
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
                'multi_mode_control_msgs.srv.SetRecoveryMode_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_recovery_mode__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_recovery_mode__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_recovery_mode__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_recovery_mode__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_recovery_mode__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'AUTO': cls.__constants['AUTO'],
            'MANUAL': cls.__constants['MANUAL'],
            'PROGRAM': cls.__constants['PROGRAM'],
        }

    @property
    def AUTO(self):
        """Message constant 'AUTO'."""
        return Metaclass_SetRecoveryMode_Request.__constants['AUTO']

    @property
    def MANUAL(self):
        """Message constant 'MANUAL'."""
        return Metaclass_SetRecoveryMode_Request.__constants['MANUAL']

    @property
    def PROGRAM(self):
        """Message constant 'PROGRAM'."""
        return Metaclass_SetRecoveryMode_Request.__constants['PROGRAM']


class SetRecoveryMode_Request(metaclass=Metaclass_SetRecoveryMode_Request):
    """
    Message class 'SetRecoveryMode_Request'.

    Constants:
      AUTO
      MANUAL
      PROGRAM
    """

    __slots__ = [
        '_mode',
        '_timeout',
    ]

    _fields_and_field_types = {
        'mode': 'int32',
        'timeout': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.mode = kwargs.get('mode', int())
        self.timeout = kwargs.get('timeout', float())

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
        if self.mode != other.mode:
            return False
        if self.timeout != other.timeout:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def mode(self):
        """Message field 'mode'."""
        return self._mode

    @mode.setter
    def mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mode' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'mode' field must be an integer in [-2147483648, 2147483647]"
        self._mode = value

    @builtins.property
    def timeout(self):
        """Message field 'timeout'."""
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'timeout' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'timeout' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._timeout = value


# Import statements for member types

# already imported above
# import rosidl_parser.definition


class Metaclass_SetRecoveryMode_Response(type):
    """Metaclass of message 'SetRecoveryMode_Response'."""

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
                'multi_mode_control_msgs.srv.SetRecoveryMode_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_recovery_mode__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_recovery_mode__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_recovery_mode__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_recovery_mode__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_recovery_mode__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetRecoveryMode_Response(metaclass=Metaclass_SetRecoveryMode_Response):
    """Message class 'SetRecoveryMode_Response'."""

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


class Metaclass_SetRecoveryMode(type):
    """Metaclass of service 'SetRecoveryMode'."""

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
                'multi_mode_control_msgs.srv.SetRecoveryMode')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__set_recovery_mode

            from multi_mode_control_msgs.srv import _set_recovery_mode
            if _set_recovery_mode.Metaclass_SetRecoveryMode_Request._TYPE_SUPPORT is None:
                _set_recovery_mode.Metaclass_SetRecoveryMode_Request.__import_type_support__()
            if _set_recovery_mode.Metaclass_SetRecoveryMode_Response._TYPE_SUPPORT is None:
                _set_recovery_mode.Metaclass_SetRecoveryMode_Response.__import_type_support__()


class SetRecoveryMode(metaclass=Metaclass_SetRecoveryMode):
    from multi_mode_control_msgs.srv._set_recovery_mode import SetRecoveryMode_Request as Request
    from multi_mode_control_msgs.srv._set_recovery_mode import SetRecoveryMode_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
