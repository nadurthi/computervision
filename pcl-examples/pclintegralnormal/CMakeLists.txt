cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

SET (PROJECTNAME pclintegralimagenormal)
project(${PROJECTNAME})

# The version number.
set (customvisualizer_VERSION_MAJOR 1)
set (customvisualizer_VERSION_MINOR 0)

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


add_executable (${PROJECTNAME} pclintegralimagenormal.cpp)
target_link_libraries (${PROJECTNAME} ${PCL_LIBRARIES})


set(EXECUTABLE_OUTPUT_PATH "../debug")