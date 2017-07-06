cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cloud_viewer)
# The version number.
set (cloud_viewer_VERSION_MAJOR 1)
set (cloud_viewer_VERSION_MINOR 0)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/cloud_viewerConfig.h.in"
  "${PROJECT_BINARY_DIR}/cloud_viewerConfig.h"
  )

# add the binary tree to the search path for include files
# so that we will find cloud_viewerConfig.h
include_directories("${PROJECT_BINARY_DIR}")


find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (cloud_viewer cloud_viewer.cpp)
target_link_libraries (cloud_viewer ${PCL_LIBRARIES})
