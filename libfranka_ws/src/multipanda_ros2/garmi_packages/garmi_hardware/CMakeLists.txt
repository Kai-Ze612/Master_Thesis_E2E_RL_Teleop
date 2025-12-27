cmake_minimum_required(VERSION 3.8)
project(garmi_hardware)


# Default to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to release
set(default_build_type "Release")

# Respect the build type from user and ide
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)

add_library(garmi_hardware
        SHARED
        src/udp_server.cpp
        src/helper_functions.cpp
        src/garmi_head_hardware_interface.cpp
        src/garmi_base_hardware_interface.cpp
)

target_include_directories(
        garmi_hardware
        PRIVATE
        include
)
ament_target_dependencies(
        garmi_hardware
        hardware_interface
        pluginlib
        rclcpp
        rclcpp_action
        rclcpp_components
)

pluginlib_export_plugin_description_file(hardware_interface garmi_hardware.xml)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install scripts
install(
        TARGETS garmi_hardware
        DESTINATION lib
)
install(
        DIRECTORY include/
        DESTINATION include
)
ament_export_include_directories(
        include
)
ament_export_libraries(
        garmi_hardware
)
ament_export_dependencies(
        hardware_interface
        pluginlib
        rclcpp
)

ament_package()
