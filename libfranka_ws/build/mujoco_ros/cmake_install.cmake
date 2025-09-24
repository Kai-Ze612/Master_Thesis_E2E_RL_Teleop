# Install script for directory: /media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/mujoco_ros_pkgs/mujoco_ros

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/install/mujoco_ros")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/include/mujoco_ros/render_backend.hpp")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/include/mujoco_ros" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/include/mujoco_ros/render_backend.hpp")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/include/mujoco_ros/ros_version.hpp")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/include/mujoco_ros" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/include/mujoco_ros/ros_version.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/" TYPE DIRECTORY FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/mujoco_ros_pkgs/mujoco_ros/include/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mujoco_ros" TYPE FILE FILES
    "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/include/mujoco_ros/ros_version.hpp"
    "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/include/mujoco_ros/render_backend.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros" TYPE DIRECTORY FILES
    "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/mujoco_ros_pkgs/mujoco_ros/launch/ros2/launch"
    "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/mujoco_ros_pkgs/mujoco_ros/config"
    "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/mujoco_ros_pkgs/mujoco_ros/assets"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmujoco_ros.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmujoco_ros.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmujoco_ros.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/src/libmujoco_ros.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmujoco_ros.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmujoco_ros.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmujoco_ros.so"
         OLD_RPATH "/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/install/mujoco_ros_msgs/lib:/home/kai/Libraries/mujoco/mujoco_install/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmujoco_ros.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/mujoco_ros/mujoco_node" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/mujoco_ros/mujoco_node")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/mujoco_ros/mujoco_node"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/mujoco_ros" TYPE EXECUTABLE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/src/mujoco_node")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/mujoco_ros/mujoco_node" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/mujoco_ros/mujoco_node")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/mujoco_ros/mujoco_node"
         OLD_RPATH "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/src:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/install/mujoco_ros_msgs/lib:/home/kai/Libraries/mujoco/mujoco_install/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/mujoco_ros/mujoco_node")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/mujoco_ros")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/mujoco_ros")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros/environment" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros/environment" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_environment_hooks/path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_environment_hooks/local_setup.bash")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_environment_hooks/local_setup.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_environment_hooks/package.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_index/share/ament_index/resource_index/packages/mujoco_ros")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros/cmake" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_export_include_directories/ament_cmake_export_include_directories-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros/cmake" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_export_dependencies/ament_cmake_export_dependencies-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros/cmake" TYPE FILE FILES
    "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_core/mujoco_rosConfig.cmake"
    "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/ament_cmake_core/mujoco_rosConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mujoco_ros" TYPE FILE FILES "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/src/mujoco_ros_pkgs/mujoco_ros/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/build/mujoco_ros/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
