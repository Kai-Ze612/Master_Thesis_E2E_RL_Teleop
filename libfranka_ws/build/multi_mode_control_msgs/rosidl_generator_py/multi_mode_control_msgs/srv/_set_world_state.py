# generated from rosidl_generator_py/resource/_idl.py.em
# with input from multi_mode_control_msgs:srv/SetWorldState.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'items'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SetWorldState_Request(type):
    """Metaclass of message 'SetWorldState_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'HUMAN_PRESENT': 0,
        'OBJECT_IN_HAND': 1,
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
                'multi_mode_control_msgs.srv.SetWorldState_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_world_state__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_world_state__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_world_state__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_world_state__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_world_state__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'HUMAN_PRESENT': cls.__constants['HUMAN_PRESENT'],
            'OBJECT_IN_HAND': cls.__constants['OBJECT_IN_HAND'],
        }

    @property
    def HUMAN_PRESENT(self):
        """Message constant 'HUMAN_PRESENT'."""
        return Metaclass_SetWorldState_Request.__constants['HUMAN_PRESENT']

    @property
    def OBJECT_IN_HAND(self):
        """Message constant 'OBJECT_IN_HAND'."""
        return Metaclass_SetWorldState_Request.__constants['OBJECT_IN_HAND']


class SetWorldState_Request(metaclass=Metaclass_SetWorldState_Request):
    """
    Message class 'SetWorldState_Request'.

    Constants:
      HUMAN_PRESENT
      OBJECT_IN_HAND
    """

    __slots__ = [
        '_items',
        '_values',
    ]

    _fields_and_field_types = {
        'items': 'sequence<int32>',
        'values': 'sequence<boolean>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int32')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('boolean')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.items = array.array('i', kwargs.get('items', []))
        self.values = kwargs.get('values', [])

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
        if self.items != other.items:
            return False
        if self.values != other.values:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def items(self):
        """Message field 'items'."""
        return self._items

    @items.setter
    def items(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'i', \
                "The 'items' array.array() must have the type code of 'i'"
            self._items = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= -2147483648 and val < 2147483648 for val in value)), \
                "The 'items' field must be a set or sequence and each value of type 'int' and each integer in [-2147483648, 2147483647]"
        self._items = array.array('i', value)

    @builtins.property
    def values(self):
        """Message field 'values'."""
        return self._values

    @values.setter
    def values(self, value):
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
                 all(isinstance(v, bool) for v in value) and
                 True), \
                "The 'values' field must be a set or sequence and each value of type 'bool'"
        self._values = value


# Import statements for member types

# already imported above
# import rosidl_parser.definition


class Metaclass_SetWorldState_Response(type):
    """Metaclass of message 'SetWorldState_Response'."""

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
                'multi_mode_control_msgs.srv.SetWorldState_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_world_state__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_world_state__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_world_state__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_world_state__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_world_state__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetWorldState_Response(metaclass=Metaclass_SetWorldState_Response):
    """Message class 'SetWorldState_Response'."""

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


class Metaclass_SetWorldState(type):
    """Metaclass of service 'SetWorldState'."""

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
                'multi_mode_control_msgs.srv.SetWorldState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__set_world_state

            from multi_mode_control_msgs.srv import _set_world_state
            if _set_world_state.Metaclass_SetWorldState_Request._TYPE_SUPPORT is None:
                _set_world_state.Metaclass_SetWorldState_Request.__import_type_support__()
            if _set_world_state.Metaclass_SetWorldState_Response._TYPE_SUPPORT is None:
                _set_world_state.Metaclass_SetWorldState_Response.__import_type_support__()


class SetWorldState(metaclass=Metaclass_SetWorldState):
    from multi_mode_control_msgs.srv._set_world_state import SetWorldState_Request as Request
    from multi_mode_control_msgs.srv._set_world_state import SetWorldState_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
