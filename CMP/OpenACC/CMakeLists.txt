################################################################################
##
#  @file CMakeLists.txt
#  @date 2017-02-20
#  @author Tiago Lobato Gimenes    (tlgimenes@gmail.com)
## 
################################################################################
# Specify the minimum version for CMake
cmake_minimum_required(VERSION 2.8)

################################################################################
# Project's name
project(cmp-acc)

################################################################################
# Set compiler
find_program(PGCC_FOUND NAMES pgcc pgpp)
if(PGCC_FOUND)
  set(CMAKE_C_COMPILER pgcc)
  set(CMAKE_CXX_COMPILER pgc++)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium -ta=nvidia:cc60 -ta=nvidia:cc50 -ta=nvidia:cc35 -ta=nvidia:cc30 -ta=multicore -Minfo=accel --c++11")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ta=radeon:hawaii -Minfo=accel --c++11")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenacc -foffload=-lm -ftarget=nvptx-none")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pg")
endif()

################################################################################
# Set CMAKE's variables
set(CMAKE_PROJECT_DIR ${CMAKE_SOURCE_DIR}/src)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(DESTDIR ${CMAKE_BINARY_DIR})

# Legacy cmake 
if(CMAKE_MAJOR_VERSION LESS 3)
  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
endif()

################################################################################
# The following folder will be included
include_directories(${PROJECT_SOURCE_DIR}/include)

################################################################################
# Add executables 
add_executable(cmp-acc
  ${CMAKE_PROJECT_DIR}/log.cpp
  ${CMAKE_PROJECT_DIR}/su_trace.cpp
  ${CMAKE_PROJECT_DIR}/su_gather.cpp
  ${CMAKE_PROJECT_DIR}/su_cdp.cpp
  ${CMAKE_PROJECT_DIR}/parser.cpp
  ${CMAKE_PROJECT_DIR}/main.cpp)

################################################################################
# Link project to system's libraries
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "") # disable gcc's -rdynamic
target_link_libraries(cmp-acc  m)

################################################################################
