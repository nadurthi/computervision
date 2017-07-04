cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(rotatecloud)
# The version number.
set (rotatecloud_VERSION_MAJOR 1)
set (rotatecloud_VERSION_MINOR 0)

# configure a header file to pass some of the CMake settings
# to the source code
# configure_file (
#   "${PROJECT_SOURCE_DIR}/cloud_viewerConfig.h.in"
#   "${PROJECT_BINARY_DIR}/cloud_viewerConfig.h"
#   )

# add the binary tree to the search path for include files
# so that we will find cloud_viewerConfig.h
include_directories("${PROJECT_BINARY_DIR}")


find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(testww OBJECT test.cpp)

add_executable (rotatecloud pclvisualize.cpp $<TARGET_OBJECTS:testww>)
target_link_libraries (rotatecloud ${PCL_LIBRARIES})


set(EXECUTABLE_OUTPUT_PATH "../debug")