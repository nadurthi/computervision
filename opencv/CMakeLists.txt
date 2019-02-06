cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

FILE (GLOB example_SRCS  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
message("-------------- examples -------------")
message(${CMAKE_CURRENT_SOURCE_DIR})
message(${example_SRCS})
message("======================================")



FOREACH (T_FILE ${example_SRCS})
	GET_FILENAME_COMPONENT (T_NAME ${T_FILE} NAME_WE)
	# project( ${T_NAME} )
	ADD_EXECUTABLE (${T_NAME} ${T_FILE})
	# add_executable( SIFTtest testSIFT.cpp )
	# target_include_directories(${T_NAME} ${OpenCV_LIBS})
	TARGET_LINK_LIBRARIES(${T_NAME} ${OpenCV_LIBS})
	# target_link_libraries( SIFTtest ${OpenCV_LIBS} )
	set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_LIST_DIR}")
ENDFOREACH (T_FILE ${example_SRCS})
