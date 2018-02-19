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
project(cmp-cuda)

################################################################################
# Set compiler
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "-std=c++11 -arch=compute_50 --gpu-code=sm_50 ${CUDA_NVCC_FLAGS}")

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
include_directories(${CUDA_INCLUDE_DIRS})

################################################################################
# Add executables 
cuda_add_executable(cmp-cuda
  ${CMAKE_PROJECT_DIR}/log.cpp
  ${CMAKE_PROJECT_DIR}/cuda.cpp
  ${CMAKE_PROJECT_DIR}/su_trace.cpp
  ${CMAKE_PROJECT_DIR}/su_gather.cpp
  ${CMAKE_PROJECT_DIR}/su_cdp.cpp
  ${CMAKE_PROJECT_DIR}/parser.cpp
  ${CMAKE_PROJECT_DIR}/main.cu)

################################################################################
# Link project to system's libraries
target_link_libraries(cmp-cuda  m ${CUDA_LIBRARIES})

################################################################################
