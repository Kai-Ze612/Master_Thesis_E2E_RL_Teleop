// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from mujoco_ros_msgs:srv/GetGeomProperties.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "mujoco_ros_msgs/srv/detail/get_geom_properties__struct.h"
#include "mujoco_ros_msgs/srv/detail/get_geom_properties__type_support.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace mujoco_ros_msgs
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _GetGeomProperties_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GetGeomProperties_Request_type_support_ids_t;

static const _GetGeomProperties_Request_type_support_ids_t _GetGeomProperties_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _GetGeomProperties_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _GetGeomProperties_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _GetGeomProperties_Request_type_support_symbol_names_t _GetGeomProperties_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mujoco_ros_msgs, srv, GetGeomProperties_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetGeomProperties_Request)),
  }
};

typedef struct _GetGeomProperties_Request_type_support_data_t
{
  void * data[2];
} _GetGeomProperties_Request_type_support_data_t;

static _GetGeomProperties_Request_type_support_data_t _GetGeomProperties_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _GetGeomProperties_Request_message_typesupport_map = {
  2,
  "mujoco_ros_msgs",
  &_GetGeomProperties_Request_message_typesupport_ids.typesupport_identifier[0],
  &_GetGeomProperties_Request_message_typesupport_symbol_names.symbol_name[0],
  &_GetGeomProperties_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t GetGeomProperties_Request_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GetGeomProperties_Request_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace mujoco_ros_msgs

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, mujoco_ros_msgs, srv, GetGeomProperties_Request)() {
  return &::mujoco_ros_msgs::srv::rosidl_typesupport_c::GetGeomProperties_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/get_geom_properties__struct.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/get_geom_properties__type_support.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace mujoco_ros_msgs
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _GetGeomProperties_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GetGeomProperties_Response_type_support_ids_t;

static const _GetGeomProperties_Response_type_support_ids_t _GetGeomProperties_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _GetGeomProperties_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _GetGeomProperties_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _GetGeomProperties_Response_type_support_symbol_names_t _GetGeomProperties_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mujoco_ros_msgs, srv, GetGeomProperties_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetGeomProperties_Response)),
  }
};

typedef struct _GetGeomProperties_Response_type_support_data_t
{
  void * data[2];
} _GetGeomProperties_Response_type_support_data_t;

static _GetGeomProperties_Response_type_support_data_t _GetGeomProperties_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _GetGeomProperties_Response_message_typesupport_map = {
  2,
  "mujoco_ros_msgs",
  &_GetGeomProperties_Response_message_typesupport_ids.typesupport_identifier[0],
  &_GetGeomProperties_Response_message_typesupport_symbol_names.symbol_name[0],
  &_GetGeomProperties_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t GetGeomProperties_Response_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GetGeomProperties_Response_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace mujoco_ros_msgs

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, mujoco_ros_msgs, srv, GetGeomProperties_Response)() {
  return &::mujoco_ros_msgs::srv::rosidl_typesupport_c::GetGeomProperties_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "mujoco_ros_msgs/srv/detail/get_geom_properties__type_support.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/service_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace mujoco_ros_msgs
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _GetGeomProperties_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GetGeomProperties_type_support_ids_t;

static const _GetGeomProperties_type_support_ids_t _GetGeomProperties_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _GetGeomProperties_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _GetGeomProperties_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _GetGeomProperties_type_support_symbol_names_t _GetGeomProperties_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mujoco_ros_msgs, srv, GetGeomProperties)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mujoco_ros_msgs, srv, GetGeomProperties)),
  }
};

typedef struct _GetGeomProperties_type_support_data_t
{
  void * data[2];
} _GetGeomProperties_type_support_data_t;

static _GetGeomProperties_type_support_data_t _GetGeomProperties_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _GetGeomProperties_service_typesupport_map = {
  2,
  "mujoco_ros_msgs",
  &_GetGeomProperties_service_typesupport_ids.typesupport_identifier[0],
  &_GetGeomProperties_service_typesupport_symbol_names.symbol_name[0],
  &_GetGeomProperties_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t GetGeomProperties_service_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GetGeomProperties_service_typesupport_map),
  rosidl_typesupport_c__get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace mujoco_ros_msgs

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_c, mujoco_ros_msgs, srv, GetGeomProperties)() {
  return &::mujoco_ros_msgs::srv::rosidl_typesupport_c::GetGeomProperties_service_type_support_handle;
}

#ifdef __cplusplus
}
#endif
