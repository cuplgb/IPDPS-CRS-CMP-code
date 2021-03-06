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
project(cmp-omp2)

################################################################################
# Find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
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

# C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

################################################################################
# The following folder will be included
include_directories(${PROJECT_SOURCE_DIR}/include)

################################################################################
# Add executables 
add_executable(cmp-omp2
  ${CMAKE_PROJECT_DIR}/log.cpp
  ${CMAKE_PROJECT_DIR}/su_trace.cpp
  ${CMAKE_PROJECT_DIR}/su_gather.cpp
  ${CMAKE_PROJECT_DIR}/su_cdp.cpp
  ${CMAKE_PROJECT_DIR}/parser.cpp
  ${CMAKE_PROJECT_DIR}/main.cpp)

################################################################################
# Link project to system's libraries
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "") # disable gcc's -rdynamic
target_link_libraries(cmp-omp2  m)

################################################################################
