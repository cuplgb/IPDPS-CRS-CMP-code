////////////////////////////////////////////////////////////////////////////////
/**
 * @file utils.cpp
 * @date 2017-03-04
 * @author Tiago Lobato Gimenes    (tlgimenes@gmail.com)
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
////////////////////////////////////////////////////////////////////////////////

#include "utils.hpp"

#include "log.hpp"

#include <iostream>
#include <sstream>
#include <fstream>

////////////////////////////////////////////////////////////////////////////////

#define CASE(ca) \
  case ca: \
    logger::log(FAIL, std::string(file) + ":" + std::to_string(line) + ": " + #ca); \
    break

void _cl_safe(cl_int err, const char* file, int line) {
  switch(err) {
    case CL_SUCCESS: break;
    CASE( CL_DEVICE_NOT_FOUND                );
    CASE( CL_DEVICE_NOT_AVAILABLE            );
    CASE( CL_COMPILER_NOT_AVAILABLE          );
    CASE( CL_MEM_OBJECT_ALLOCATION_FAILURE   );
    CASE( CL_OUT_OF_RESOURCES                );
    CASE( CL_OUT_OF_HOST_MEMORY              );
    CASE( CL_PROFILING_INFO_NOT_AVAILABLE    );
    CASE( CL_MEM_COPY_OVERLAP                );
    CASE( CL_IMAGE_FORMAT_MISMATCH           );
    CASE( CL_IMAGE_FORMAT_NOT_SUPPORTED      );
    CASE( CL_BUILD_PROGRAM_FAILURE           );
    CASE( CL_MAP_FAILURE                     );
    CASE( CL_INVALID_VALUE                   );
    CASE( CL_INVALID_DEVICE_TYPE             );
    CASE( CL_INVALID_PLATFORM                );
    CASE( CL_INVALID_DEVICE                  );
    CASE( CL_INVALID_CONTEXT                 );
    CASE( CL_INVALID_QUEUE_PROPERTIES        );
    CASE( CL_INVALID_COMMAND_QUEUE           );
    CASE( CL_INVALID_HOST_PTR                );
    CASE( CL_INVALID_MEM_OBJECT              );
    CASE( CL_INVALID_IMAGE_FORMAT_DESCRIPTOR );
    CASE( CL_INVALID_IMAGE_SIZE              );
    CASE( CL_INVALID_SAMPLER                 );
    CASE( CL_INVALID_BINARY                  );
    CASE( CL_INVALID_BUILD_OPTIONS           );
    CASE( CL_INVALID_PROGRAM                 );
    CASE( CL_INVALID_PROGRAM_EXECUTABLE      );
    CASE( CL_INVALID_KERNEL_NAME             );
    CASE( CL_INVALID_KERNEL_DEFINITION       );
    CASE( CL_INVALID_KERNEL                  );
    CASE( CL_INVALID_ARG_INDEX               );
    CASE( CL_INVALID_ARG_VALUE               );
    CASE( CL_INVALID_ARG_SIZE                );
    CASE( CL_INVALID_KERNEL_ARGS             );
    CASE( CL_INVALID_WORK_DIMENSION          );
    CASE( CL_INVALID_WORK_GROUP_SIZE         );
    CASE( CL_INVALID_WORK_ITEM_SIZE          );
    CASE( CL_INVALID_GLOBAL_OFFSET           );
    CASE( CL_INVALID_EVENT_WAIT_LIST         );
    CASE( CL_INVALID_EVENT                   );
    CASE( CL_INVALID_OPERATION               );
    CASE( CL_INVALID_GL_OBJECT               );
    CASE( CL_INVALID_BUFFER_SIZE             );
    CASE( CL_INVALID_MIP_LEVEL               );
    CASE( CL_INVALID_GLOBAL_WORK_SIZE        );
    default:
      logger::log(FAIL, std::string(file) + ":" + std::to_string(line) + ": OpenCL unknown error " + std::to_string(err));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Read file as string
std::string file2str(const std::string& path) {
  std::ifstream program_file(path);
  std::stringstream program_src;
  program_src << program_file.rdbuf();

  return program_src.str();
}

////////////////////////////////////////////////////////////////////////////////

#define to_os(cl_device_info)\
  os << #cl_device_info << ": " << dev.getInfo<cl_device_info>() << std::endl;

std::ostream& cl::operator<<(std::ostream& os, cl::Device& dev)
{
  to_os(CL_DEVICE_PLATFORM)
  to_os(CL_DEVICE_OPENCL_C_VERSION)
  to_os(CL_DEVICE_VENDOR)
  to_os(CL_DEVICE_VENDOR_ID)
  to_os(CL_DEVICE_NAME)
  to_os(CL_DEVICE_TYPE)
  to_os(CL_DRIVER_VERSION)
  to_os(CL_DEVICE_PROFILE)
  to_os(CL_DEVICE_AVAILABLE)
  to_os(CL_DEVICE_COMPILER_AVAILABLE)
  to_os(CL_DEVICE_EXECUTION_CAPABILITIES)
  to_os(CL_DEVICE_QUEUE_PROPERTIES)
  to_os(CL_DEVICE_EXTENSIONS)
  to_os(CL_DEVICE_MAX_COMPUTE_UNITS)
  to_os(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)
  to_os(CL_DEVICE_MAX_WORK_GROUP_SIZE)
  //to_os(CL_DEVICE_MAX_WORK_ITEM_SIZES)
  to_os(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR)
  to_os(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT)
  to_os(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT)
  to_os(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG)
  to_os(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)
  to_os(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE)
  to_os(CL_DEVICE_MAX_CLOCK_FREQUENCY)
  to_os(CL_DEVICE_ADDRESS_BITS)
  to_os(CL_DEVICE_MAX_READ_IMAGE_ARGS)
  to_os(CL_DEVICE_MAX_WRITE_IMAGE_ARGS)
  to_os(CL_DEVICE_MAX_MEM_ALLOC_SIZE)
  to_os(CL_DEVICE_IMAGE2D_MAX_WIDTH)
  to_os(CL_DEVICE_IMAGE2D_MAX_HEIGHT)
  to_os(CL_DEVICE_IMAGE3D_MAX_WIDTH)
  to_os(CL_DEVICE_IMAGE3D_MAX_HEIGHT)
  to_os(CL_DEVICE_IMAGE3D_MAX_DEPTH)
  to_os(CL_DEVICE_IMAGE_SUPPORT)
  to_os(CL_DEVICE_MAX_PARAMETER_SIZE)
  to_os(CL_DEVICE_MAX_SAMPLERS)
  to_os(CL_DEVICE_MEM_BASE_ADDR_ALIGN)
  to_os(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE)
  to_os(CL_DEVICE_SINGLE_FP_CONFIG)
  to_os(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE)
  to_os(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE)
  to_os(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)
  to_os(CL_DEVICE_GLOBAL_MEM_SIZE)
  to_os(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)
  to_os(CL_DEVICE_MAX_CONSTANT_ARGS)
  to_os(CL_DEVICE_LOCAL_MEM_TYPE)
  to_os(CL_DEVICE_LOCAL_MEM_SIZE)
  to_os(CL_DEVICE_ERROR_CORRECTION_SUPPORT)
  to_os(CL_DEVICE_PROFILING_TIMER_RESOLUTION)
  to_os(CL_DEVICE_ENDIAN_LITTLE)
  to_os(CL_DEVICE_DOUBLE_FP_CONFIG)
  to_os(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF)
  //to_os(CL_DEVICE_HOST_UNIFIED_MEMORY)
  to_os(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR)
  to_os(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT)
  to_os(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT)
  to_os(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG)
  to_os(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT)
  to_os(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE)
  to_os(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF)
  //to_os(CL_DEVICE_LINKER_AVAILABLE)
  //to_os(CL_DEVICE_BUILT_IN_KERNELS)
  //to_os(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE)
  //to_os(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE)
  //to_os(CL_DEVICE_PARENT_DEVICE)
  //to_os(CL_DEVICE_PARTITION_MAX_SUB_DEVICES)
  //to_os(CL_DEVICE_PARTITION_PROPERTIES)
  //to_os(CL_DEVICE_PARTITION_AFFINITY_DOMAIN)
  //to_os(CL_DEVICE_PARTITION_TYPE)
  //to_os(CL_DEVICE_REFERENCE_COUNT)
  //to_os(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC)
  //to_os(CL_DEVICE_PRINTF_BUFFER_SIZE)

  return os;
}

////////////////////////////////////////////////////////////////////////////////

std::ostream& cl::operator<<(std::ostream& os, const cl::Error& error)
{
  switch(error.err()){
    // run-time and JIT compiler errors
    case 0  : os << "CL_SUCCESS"; break;
    case -1 : os << "CL_DEVICE_NOT_FOUND"; break;
    case -2 : os << "CL_DEVICE_NOT_AVAILABLE"; break;
    case -3 : os << "CL_COMPILER_NOT_AVAILABLE"; break;
    case -4 : os << "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
    case -5 : os << "CL_OUT_OF_RESOURCES"; break;
    case -6 : os << "CL_OUT_OF_HOST_MEMORY"; break;
    case -7 : os << "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
    case -8 : os << "CL_MEM_COPY_OVERLAP"; break;
    case -9 : os << "CL_IMAGE_FORMAT_MISMATCH"; break;
    case -10: os << "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
    case -11: os << "CL_BUILD_PROGRAM_FAILURE"; break;
    case -12: os << "CL_MAP_FAILURE"; break;
    case -13: os << "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
    case -14: os << "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
    case -15: os << "CL_COMPILE_PROGRAM_FAILURE"; break;
    case -16: os << "CL_LINKER_NOT_AVAILABLE"; break;
    case -17: os << "CL_LINK_PROGRAM_FAILURE"; break;
    case -18: os << "CL_DEVICE_PARTITION_FAILED"; break;
    case -19: os << "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;

    // compile-time errors
    case -30: os << "CL_INVALID_VALUE"; break;
    case -31: os << "CL_INVALID_DEVICE_TYPE"; break;
    case -32: os << "CL_INVALID_PLATFORM"; break;
    case -33: os << "CL_INVALID_DEVICE"; break;
    case -34: os << "CL_INVALID_CONTEXT"; break;
    case -35: os << "CL_INVALID_QUEUE_PROPERTIES"; break;
    case -36: os << "CL_INVALID_COMMAND_QUEUE"; break;
    case -37: os << "CL_INVALID_HOST_PTR"; break;
    case -38: os << "CL_INVALID_MEM_OBJECT"; break;
    case -39: os << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
    case -40: os << "CL_INVALID_IMAGE_SIZE"; break;
    case -41: os << "CL_INVALID_SAMPLER"; break;
    case -42: os << "CL_INVALID_BINARY"; break;
    case -43: os << "CL_INVALID_BUILD_OPTIONS"; break;
    case -44: os << "CL_INVALID_PROGRAM"; break;
    case -45: os << "CL_INVALID_PROGRAM_EXECUTABLE"; break;
    case -46: os << "CL_INVALID_KERNEL_NAME"; break;
    case -47: os << "CL_INVALID_KERNEL_DEFINITION"; break;
    case -48: os << "CL_INVALID_KERNEL"; break;
    case -49: os << "CL_INVALID_ARG_INDEX"; break;
    case -50: os << "CL_INVALID_ARG_VALUE"; break;
    case -51: os << "CL_INVALID_ARG_SIZE"; break;
    case -52: os << "CL_INVALID_KERNEL_ARGS"; break;
    case -53: os << "CL_INVALID_WORK_DIMENSION"; break;
    case -54: os << "CL_INVALID_WORK_GROUP_SIZE"; break;
    case -55: os << "CL_INVALID_WORK_ITEM_SIZE"; break;
    case -56: os << "CL_INVALID_GLOBAL_OFFSET"; break;
    case -57: os << "CL_INVALID_EVENT_WAIT_LIST"; break;
    case -58: os << "CL_INVALID_EVENT"; break;
    case -59: os << "CL_INVALID_OPERATION"; break;
    case -60: os << "CL_INVALID_GL_OBJECT"; break;
    case -61: os << "CL_INVALID_BUFFER_SIZE"; break;
    case -62: os << "CL_INVALID_MIP_LEVEL"; break;
    case -63: os << "CL_INVALID_GLOBAL_WORK_SIZE"; break;
    case -64: os << "CL_INVALID_PROPERTY"; break;
    case -65: os << "CL_INVALID_IMAGE_DESCRIPTOR"; break;
    case -66: os << "CL_INVALID_COMPILER_OPTIONS"; break;
    case -67: os << "CL_INVALID_LINKER_OPTIONS"; break;
    case -68: os << "CL_INVALID_DEVICE_PARTITION_COUNT"; break;

    // extension errors
    case -1000: os << "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR"; break;
    case -1001: os << "CL_PLATFORM_NOT_FOUND_KHR"; break;
    case -1002: os << "CL_INVALID_D3D10_DEVICE_KHR"; break;
    case -1003: os << "CL_INVALID_D3D10_RESOURCE_KHR"; break;
    case -1004: os << "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR"; break;
    case -1005: os << "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR"; break;
    default: os << "Unknown OpenCL error " << std::to_string(error.err());
  }

  return os;
}

////////////////////////////////////////////////////////////////////////////////
