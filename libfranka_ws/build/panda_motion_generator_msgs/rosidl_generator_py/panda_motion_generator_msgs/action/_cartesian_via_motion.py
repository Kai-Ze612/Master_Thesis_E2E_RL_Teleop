# generated from rosidl_generator_py/resource/_idl.py.em
# with input from panda_motion_generator_msgs:action/CartesianViaMotion.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CartesianViaMotion_Goal(type):
    """Metaclass of message 'CartesianViaMotion_Goal'."""

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
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_Goal')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__action__cartesian_via_motion__goal
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__action__cartesian_via_motion__goal
            cls._CONVERT_TO_PY = module.convert_to_py_msg__action__cartesian_via_motion__goal
            cls._TYPE_SUPPORT = module.type_support_msg__action__cartesian_via_motion__goal
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__action__cartesian_via_motion__goal

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


class CartesianViaMotion_Goal(metaclass=Metaclass_CartesianViaMotion_Goal):
    """Message class 'CartesianViaMotion_Goal'."""

    __slots__ = [
        '_via_poses',
        '_v_scale',
    ]

    _fields_and_field_types = {
        'via_poses': 'sequence<geometry_msgs/Pose>',
        'v_scale': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Pose')),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.via_poses = kwargs.get('via_poses', [])
        self.v_scale = kwargs.get('v_scale', float())

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
        if self.via_poses != other.via_poses:
            return False
        if self.v_scale != other.v_scale:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def via_poses(self):
        """Message field 'via_poses'."""
        return self._via_poses

    @via_poses.setter
    def via_poses(self, value):
        if __debug__:
            from geometry_msgs.msg import Pose
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
                 all(isinstance(v, Pose) for v in value) and
                 True), \
                "The 'via_poses' field must be a set or sequence and each value of type 'Pose'"
        self._via_poses = value

    @builtins.property
    def v_scale(self):
        """Message field 'v_scale'."""
        return self._v_scale

    @v_scale.setter
    def v_scale(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'v_scale' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'v_scale' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._v_scale = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_CartesianViaMotion_Result(type):
    """Metaclass of message 'CartesianViaMotion_Result'."""

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
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_Result')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__action__cartesian_via_motion__result
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__action__cartesian_via_motion__result
            cls._CONVERT_TO_PY = module.convert_to_py_msg__action__cartesian_via_motion__result
            cls._TYPE_SUPPORT = module.type_support_msg__action__cartesian_via_motion__result
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__action__cartesian_via_motion__result

            from panda_motion_generator_msgs.msg import SimpleActionResult
            if SimpleActionResult.__class__._TYPE_SUPPORT is None:
                SimpleActionResult.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CartesianViaMotion_Result(metaclass=Metaclass_CartesianViaMotion_Result):
    """Message class 'CartesianViaMotion_Result'."""

    __slots__ = [
        '_result',
    ]

    _fields_and_field_types = {
        'result': 'panda_motion_generator_msgs/SimpleActionResult',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['panda_motion_generator_msgs', 'msg'], 'SimpleActionResult'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from panda_motion_generator_msgs.msg import SimpleActionResult
        self.result = kwargs.get('result', SimpleActionResult())

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
        if self.result != other.result:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def result(self):
        """Message field 'result'."""
        return self._result

    @result.setter
    def result(self, value):
        if __debug__:
            from panda_motion_generator_msgs.msg import SimpleActionResult
            assert \
                isinstance(value, SimpleActionResult), \
                "The 'result' field must be a sub message of type 'SimpleActionResult'"
        self._result = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import math

# already imported above
# import rosidl_parser.definition


class Metaclass_CartesianViaMotion_Feedback(type):
    """Metaclass of message 'CartesianViaMotion_Feedback'."""

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
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_Feedback')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__action__cartesian_via_motion__feedback
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__action__cartesian_via_motion__feedback
            cls._CONVERT_TO_PY = module.convert_to_py_msg__action__cartesian_via_motion__feedback
            cls._TYPE_SUPPORT = module.type_support_msg__action__cartesian_via_motion__feedback
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__action__cartesian_via_motion__feedback

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CartesianViaMotion_Feedback(metaclass=Metaclass_CartesianViaMotion_Feedback):
    """Message class 'CartesianViaMotion_Feedback'."""

    __slots__ = [
        '_progress',
        '_time_to_completion',
    ]

    _fields_and_field_types = {
        'progress': 'float',
        'time_to_completion': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.progress = kwargs.get('progress', float())
        self.time_to_completion = kwargs.get('time_to_completion', float())

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
        if self.progress != other.progress:
            return False
        if self.time_to_completion != other.time_to_completion:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def progress(self):
        """Message field 'progress'."""
        return self._progress

    @progress.setter
    def progress(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'progress' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'progress' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._progress = value

    @builtins.property
    def time_to_completion(self):
        """Message field 'time_to_completion'."""
        return self._time_to_completion

    @time_to_completion.setter
    def time_to_completion(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'time_to_completion' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'time_to_completion' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._time_to_completion = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_CartesianViaMotion_SendGoal_Request(type):
    """Metaclass of message 'CartesianViaMotion_SendGoal_Request'."""

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
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_SendGoal_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__action__cartesian_via_motion__send_goal__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__action__cartesian_via_motion__send_goal__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__action__cartesian_via_motion__send_goal__request
            cls._TYPE_SUPPORT = module.type_support_msg__action__cartesian_via_motion__send_goal__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__action__cartesian_via_motion__send_goal__request

            from panda_motion_generator_msgs.action import CartesianViaMotion
            if CartesianViaMotion.Goal.__class__._TYPE_SUPPORT is None:
                CartesianViaMotion.Goal.__class__.__import_type_support__()

            from unique_identifier_msgs.msg import UUID
            if UUID.__class__._TYPE_SUPPORT is None:
                UUID.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CartesianViaMotion_SendGoal_Request(metaclass=Metaclass_CartesianViaMotion_SendGoal_Request):
    """Message class 'CartesianViaMotion_SendGoal_Request'."""

    __slots__ = [
        '_goal_id',
        '_goal',
    ]

    _fields_and_field_types = {
        'goal_id': 'unique_identifier_msgs/UUID',
        'goal': 'panda_motion_generator_msgs/CartesianViaMotion_Goal',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['unique_identifier_msgs', 'msg'], 'UUID'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['panda_motion_generator_msgs', 'action'], 'CartesianViaMotion_Goal'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from unique_identifier_msgs.msg import UUID
        self.goal_id = kwargs.get('goal_id', UUID())
        from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_Goal
        self.goal = kwargs.get('goal', CartesianViaMotion_Goal())

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
        if self.goal_id != other.goal_id:
            return False
        if self.goal != other.goal:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def goal_id(self):
        """Message field 'goal_id'."""
        return self._goal_id

    @goal_id.setter
    def goal_id(self, value):
        if __debug__:
            from unique_identifier_msgs.msg import UUID
            assert \
                isinstance(value, UUID), \
                "The 'goal_id' field must be a sub message of type 'UUID'"
        self._goal_id = value

    @builtins.property
    def goal(self):
        """Message field 'goal'."""
        return self._goal

    @goal.setter
    def goal(self, value):
        if __debug__:
            from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_Goal
            assert \
                isinstance(value, CartesianViaMotion_Goal), \
                "The 'goal' field must be a sub message of type 'CartesianViaMotion_Goal'"
        self._goal = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_CartesianViaMotion_SendGoal_Response(type):
    """Metaclass of message 'CartesianViaMotion_SendGoal_Response'."""

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
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_SendGoal_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__action__cartesian_via_motion__send_goal__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__action__cartesian_via_motion__send_goal__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__action__cartesian_via_motion__send_goal__response
            cls._TYPE_SUPPORT = module.type_support_msg__action__cartesian_via_motion__send_goal__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__action__cartesian_via_motion__send_goal__response

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CartesianViaMotion_SendGoal_Response(metaclass=Metaclass_CartesianViaMotion_SendGoal_Response):
    """Message class 'CartesianViaMotion_SendGoal_Response'."""

    __slots__ = [
        '_accepted',
        '_stamp',
    ]

    _fields_and_field_types = {
        'accepted': 'boolean',
        'stamp': 'builtin_interfaces/Time',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.accepted = kwargs.get('accepted', bool())
        from builtin_interfaces.msg import Time
        self.stamp = kwargs.get('stamp', Time())

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
        if self.accepted != other.accepted:
            return False
        if self.stamp != other.stamp:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def accepted(self):
        """Message field 'accepted'."""
        return self._accepted

    @accepted.setter
    def accepted(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'accepted' field must be of type 'bool'"
        self._accepted = value

    @builtins.property
    def stamp(self):
        """Message field 'stamp'."""
        return self._stamp

    @stamp.setter
    def stamp(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'stamp' field must be a sub message of type 'Time'"
        self._stamp = value


class Metaclass_CartesianViaMotion_SendGoal(type):
    """Metaclass of service 'CartesianViaMotion_SendGoal'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_SendGoal')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__action__cartesian_via_motion__send_goal

            from panda_motion_generator_msgs.action import _cartesian_via_motion
            if _cartesian_via_motion.Metaclass_CartesianViaMotion_SendGoal_Request._TYPE_SUPPORT is None:
                _cartesian_via_motion.Metaclass_CartesianViaMotion_SendGoal_Request.__import_type_support__()
            if _cartesian_via_motion.Metaclass_CartesianViaMotion_SendGoal_Response._TYPE_SUPPORT is None:
                _cartesian_via_motion.Metaclass_CartesianViaMotion_SendGoal_Response.__import_type_support__()


class CartesianViaMotion_SendGoal(metaclass=Metaclass_CartesianViaMotion_SendGoal):
    from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_SendGoal_Request as Request
    from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_SendGoal_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_CartesianViaMotion_GetResult_Request(type):
    """Metaclass of message 'CartesianViaMotion_GetResult_Request'."""

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
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_GetResult_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__action__cartesian_via_motion__get_result__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__action__cartesian_via_motion__get_result__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__action__cartesian_via_motion__get_result__request
            cls._TYPE_SUPPORT = module.type_support_msg__action__cartesian_via_motion__get_result__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__action__cartesian_via_motion__get_result__request

            from unique_identifier_msgs.msg import UUID
            if UUID.__class__._TYPE_SUPPORT is None:
                UUID.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CartesianViaMotion_GetResult_Request(metaclass=Metaclass_CartesianViaMotion_GetResult_Request):
    """Message class 'CartesianViaMotion_GetResult_Request'."""

    __slots__ = [
        '_goal_id',
    ]

    _fields_and_field_types = {
        'goal_id': 'unique_identifier_msgs/UUID',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['unique_identifier_msgs', 'msg'], 'UUID'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from unique_identifier_msgs.msg import UUID
        self.goal_id = kwargs.get('goal_id', UUID())

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
        if self.goal_id != other.goal_id:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def goal_id(self):
        """Message field 'goal_id'."""
        return self._goal_id

    @goal_id.setter
    def goal_id(self, value):
        if __debug__:
            from unique_identifier_msgs.msg import UUID
            assert \
                isinstance(value, UUID), \
                "The 'goal_id' field must be a sub message of type 'UUID'"
        self._goal_id = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_CartesianViaMotion_GetResult_Response(type):
    """Metaclass of message 'CartesianViaMotion_GetResult_Response'."""

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
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_GetResult_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__action__cartesian_via_motion__get_result__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__action__cartesian_via_motion__get_result__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__action__cartesian_via_motion__get_result__response
            cls._TYPE_SUPPORT = module.type_support_msg__action__cartesian_via_motion__get_result__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__action__cartesian_via_motion__get_result__response

            from panda_motion_generator_msgs.action import CartesianViaMotion
            if CartesianViaMotion.Result.__class__._TYPE_SUPPORT is None:
                CartesianViaMotion.Result.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CartesianViaMotion_GetResult_Response(metaclass=Metaclass_CartesianViaMotion_GetResult_Response):
    """Message class 'CartesianViaMotion_GetResult_Response'."""

    __slots__ = [
        '_status',
        '_result',
    ]

    _fields_and_field_types = {
        'status': 'int8',
        'result': 'panda_motion_generator_msgs/CartesianViaMotion_Result',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['panda_motion_generator_msgs', 'action'], 'CartesianViaMotion_Result'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.status = kwargs.get('status', int())
        from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_Result
        self.result = kwargs.get('result', CartesianViaMotion_Result())

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
        if self.status != other.status:
            return False
        if self.result != other.result:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def status(self):
        """Message field 'status'."""
        return self._status

    @status.setter
    def status(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'status' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'status' field must be an integer in [-128, 127]"
        self._status = value

    @builtins.property
    def result(self):
        """Message field 'result'."""
        return self._result

    @result.setter
    def result(self, value):
        if __debug__:
            from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_Result
            assert \
                isinstance(value, CartesianViaMotion_Result), \
                "The 'result' field must be a sub message of type 'CartesianViaMotion_Result'"
        self._result = value


class Metaclass_CartesianViaMotion_GetResult(type):
    """Metaclass of service 'CartesianViaMotion_GetResult'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_GetResult')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__action__cartesian_via_motion__get_result

            from panda_motion_generator_msgs.action import _cartesian_via_motion
            if _cartesian_via_motion.Metaclass_CartesianViaMotion_GetResult_Request._TYPE_SUPPORT is None:
                _cartesian_via_motion.Metaclass_CartesianViaMotion_GetResult_Request.__import_type_support__()
            if _cartesian_via_motion.Metaclass_CartesianViaMotion_GetResult_Response._TYPE_SUPPORT is None:
                _cartesian_via_motion.Metaclass_CartesianViaMotion_GetResult_Response.__import_type_support__()


class CartesianViaMotion_GetResult(metaclass=Metaclass_CartesianViaMotion_GetResult):
    from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_GetResult_Request as Request
    from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_GetResult_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_CartesianViaMotion_FeedbackMessage(type):
    """Metaclass of message 'CartesianViaMotion_FeedbackMessage'."""

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
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion_FeedbackMessage')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__action__cartesian_via_motion__feedback_message
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__action__cartesian_via_motion__feedback_message
            cls._CONVERT_TO_PY = module.convert_to_py_msg__action__cartesian_via_motion__feedback_message
            cls._TYPE_SUPPORT = module.type_support_msg__action__cartesian_via_motion__feedback_message
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__action__cartesian_via_motion__feedback_message

            from panda_motion_generator_msgs.action import CartesianViaMotion
            if CartesianViaMotion.Feedback.__class__._TYPE_SUPPORT is None:
                CartesianViaMotion.Feedback.__class__.__import_type_support__()

            from unique_identifier_msgs.msg import UUID
            if UUID.__class__._TYPE_SUPPORT is None:
                UUID.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CartesianViaMotion_FeedbackMessage(metaclass=Metaclass_CartesianViaMotion_FeedbackMessage):
    """Message class 'CartesianViaMotion_FeedbackMessage'."""

    __slots__ = [
        '_goal_id',
        '_feedback',
    ]

    _fields_and_field_types = {
        'goal_id': 'unique_identifier_msgs/UUID',
        'feedback': 'panda_motion_generator_msgs/CartesianViaMotion_Feedback',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['unique_identifier_msgs', 'msg'], 'UUID'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['panda_motion_generator_msgs', 'action'], 'CartesianViaMotion_Feedback'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from unique_identifier_msgs.msg import UUID
        self.goal_id = kwargs.get('goal_id', UUID())
        from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_Feedback
        self.feedback = kwargs.get('feedback', CartesianViaMotion_Feedback())

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
        if self.goal_id != other.goal_id:
            return False
        if self.feedback != other.feedback:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def goal_id(self):
        """Message field 'goal_id'."""
        return self._goal_id

    @goal_id.setter
    def goal_id(self, value):
        if __debug__:
            from unique_identifier_msgs.msg import UUID
            assert \
                isinstance(value, UUID), \
                "The 'goal_id' field must be a sub message of type 'UUID'"
        self._goal_id = value

    @builtins.property
    def feedback(self):
        """Message field 'feedback'."""
        return self._feedback

    @feedback.setter
    def feedback(self, value):
        if __debug__:
            from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_Feedback
            assert \
                isinstance(value, CartesianViaMotion_Feedback), \
                "The 'feedback' field must be a sub message of type 'CartesianViaMotion_Feedback'"
        self._feedback = value


class Metaclass_CartesianViaMotion(type):
    """Metaclass of action 'CartesianViaMotion'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('panda_motion_generator_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'panda_motion_generator_msgs.action.CartesianViaMotion')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_action__action__cartesian_via_motion

            from action_msgs.msg import _goal_status_array
            if _goal_status_array.Metaclass_GoalStatusArray._TYPE_SUPPORT is None:
                _goal_status_array.Metaclass_GoalStatusArray.__import_type_support__()
            from action_msgs.srv import _cancel_goal
            if _cancel_goal.Metaclass_CancelGoal._TYPE_SUPPORT is None:
                _cancel_goal.Metaclass_CancelGoal.__import_type_support__()

            from panda_motion_generator_msgs.action import _cartesian_via_motion
            if _cartesian_via_motion.Metaclass_CartesianViaMotion_SendGoal._TYPE_SUPPORT is None:
                _cartesian_via_motion.Metaclass_CartesianViaMotion_SendGoal.__import_type_support__()
            if _cartesian_via_motion.Metaclass_CartesianViaMotion_GetResult._TYPE_SUPPORT is None:
                _cartesian_via_motion.Metaclass_CartesianViaMotion_GetResult.__import_type_support__()
            if _cartesian_via_motion.Metaclass_CartesianViaMotion_FeedbackMessage._TYPE_SUPPORT is None:
                _cartesian_via_motion.Metaclass_CartesianViaMotion_FeedbackMessage.__import_type_support__()


class CartesianViaMotion(metaclass=Metaclass_CartesianViaMotion):

    # The goal message defined in the action definition.
    from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_Goal as Goal
    # The result message defined in the action definition.
    from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_Result as Result
    # The feedback message defined in the action definition.
    from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_Feedback as Feedback

    class Impl:

        # The send_goal service using a wrapped version of the goal message as a request.
        from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_SendGoal as SendGoalService
        # The get_result service using a wrapped version of the result message as a response.
        from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_GetResult as GetResultService
        # The feedback message with generic fields which wraps the feedback message.
        from panda_motion_generator_msgs.action._cartesian_via_motion import CartesianViaMotion_FeedbackMessage as FeedbackMessage

        # The generic service to cancel a goal.
        from action_msgs.srv._cancel_goal import CancelGoal as CancelGoalService
        # The generic message for get the status of a goal.
        from action_msgs.msg._goal_status_array import GoalStatusArray as GoalStatusMessage

    def __init__(self):
        raise NotImplementedError('Action classes can not be instantiated')
