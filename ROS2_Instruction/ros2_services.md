ros2 services are for request-response communication. it is a function calls between nodes.

ros2 services is one time request response.

structure:
client: who send request
server: response success / fail

It is used for parameter Changes.

It is a reliable communication, it will guaranteed delievery.

It is a sync operation, the client will wait until then response is receieved before continuing.

We can use ros2 service to tune the PD parameters.

## Check available:
```
# List all available services
ros2 service list

# Get information about a specific service
ros2 service info /service_server/set_cartesian_stiffness

# See what message type a service uses
ros2 service type /service_server/set_cartesian_stiffness

# See the structure of the service message
ros2 interface show franka_msgs/srv/SetCartesianStiffness
```

## Call:
```
# Basic syntax: ros2 service call <service_name> <service_type> "<message_content>"

# Example 1: Set cartesian stiffness
ros2 service call /service_server/set_cartesian_stiffness franka_msgs/srv/SetCartesianStiffness \
  "{cartesian_stiffness: [3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0]}"
```

## Response
```
# Basic syntax: ros2 service response <service_name>

# Example: Check response from the service
ros2 service response /service_server/set_cartesian_stiffness
```

