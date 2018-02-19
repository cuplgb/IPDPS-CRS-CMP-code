////////////////////////////////////////////////////////////////////////////////
/**
 * @file cuda.cpp
 * @date 2017-03-31
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

#include "cuda.hpp"

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include "log.hpp"

////////////////////////////////////////////////////////////////////////////////

#define caseToStr(c) case c: return std::string(#c)

std::string cudaErrorToString(cudaError_t err) {
	switch(err) {
		caseToStr(cudaSuccess                         );
		caseToStr(cudaErrorMissingConfiguration       );
		caseToStr(cudaErrorMemoryAllocation           );
		caseToStr(cudaErrorInitializationError        );
		caseToStr(cudaErrorLaunchFailure              );
		caseToStr(cudaErrorPriorLaunchFailure         );
		caseToStr(cudaErrorLaunchTimeout              );
		caseToStr(cudaErrorLaunchOutOfResources       );
		caseToStr(cudaErrorInvalidDeviceFunction      );
		caseToStr(cudaErrorInvalidConfiguration       );
		caseToStr(cudaErrorInvalidDevice              );
		caseToStr(cudaErrorInvalidValue               );
		caseToStr(cudaErrorInvalidPitchValue          );
		caseToStr(cudaErrorInvalidSymbol              );
		caseToStr(cudaErrorMapBufferObjectFailed      );
		caseToStr(cudaErrorUnmapBufferObjectFailed    );
		caseToStr(cudaErrorInvalidHostPointer         );
		caseToStr(cudaErrorInvalidDevicePointer       );
		caseToStr(cudaErrorInvalidTexture             );
		caseToStr(cudaErrorInvalidTextureBinding      );
		caseToStr(cudaErrorInvalidChannelDescriptor   );
		caseToStr(cudaErrorInvalidMemcpyDirection     );
		caseToStr(cudaErrorAddressOfConstant          );
		caseToStr(cudaErrorTextureFetchFailed         );
		caseToStr(cudaErrorTextureNotBound            );
		caseToStr(cudaErrorSynchronizationError       );
		caseToStr(cudaErrorInvalidFilterSetting       );
		caseToStr(cudaErrorInvalidNormSetting         );
		caseToStr(cudaErrorMixedDeviceExecution       );
		caseToStr(cudaErrorCudartUnloading            );
		caseToStr(cudaErrorUnknown                    );
		caseToStr(cudaErrorNotYetImplemented          );
		caseToStr(cudaErrorMemoryValueTooLarge        );
		caseToStr(cudaErrorInvalidResourceHandle      );
		caseToStr(cudaErrorNotReady                   );
		caseToStr(cudaErrorInsufficientDriver         );
		caseToStr(cudaErrorSetOnActiveProcess         );
		caseToStr(cudaErrorInvalidSurface             );
		caseToStr(cudaErrorNoDevice                   );
		caseToStr(cudaErrorECCUncorrectable           );
		caseToStr(cudaErrorSharedObjectSymbolNotFound );
		caseToStr(cudaErrorSharedObjectInitFailed     );
		caseToStr(cudaErrorUnsupportedLimit           );
		caseToStr(cudaErrorDuplicateVariableName      );
		caseToStr(cudaErrorDuplicateTextureName       );
		caseToStr(cudaErrorDuplicateSurfaceName       );
		caseToStr(cudaErrorDevicesUnavailable         );
		caseToStr(cudaErrorInvalidKernelImage         );
		caseToStr(cudaErrorNoKernelImageForDevice     );
		caseToStr(cudaErrorIncompatibleDriverContext  );
		caseToStr(cudaErrorPeerAccessAlreadyEnabled   );
		caseToStr(cudaErrorPeerAccessNotEnabled       );
		caseToStr(cudaErrorDeviceAlreadyInUse         );
		caseToStr(cudaErrorProfilerDisabled           );
		caseToStr(cudaErrorProfilerNotInitialized     );
		caseToStr(cudaErrorProfilerAlreadyStarted     );
		caseToStr(cudaErrorProfilerAlreadyStopped     );
		caseToStr(cudaErrorAssert                     );
		caseToStr(cudaErrorTooManyPeers               );
		caseToStr(cudaErrorHostMemoryAlreadyRegistered);
		caseToStr(cudaErrorHostMemoryNotRegistered    );
		caseToStr(cudaErrorOperatingSystem            );
		caseToStr(cudaErrorPeerAccessUnsupported      );
		caseToStr(cudaErrorLaunchMaxDepthExceeded     );
		caseToStr(cudaErrorLaunchFileScopedTex        );
		caseToStr(cudaErrorLaunchFileScopedSurf       );
		caseToStr(cudaErrorSyncDepthExceeded          );
		caseToStr(cudaErrorLaunchPendingCountExceeded );
		caseToStr(cudaErrorNotPermitted               );
		caseToStr(cudaErrorNotSupported               );
		caseToStr(cudaErrorHardwareStackError         );
		caseToStr(cudaErrorIllegalInstruction         );
		caseToStr(cudaErrorMisalignedAddress          );
		caseToStr(cudaErrorInvalidAddressSpace        );
		caseToStr(cudaErrorInvalidPc                  );
		caseToStr(cudaErrorIllegalAddress             );
		caseToStr(cudaErrorInvalidPtx                 );
		caseToStr(cudaErrorInvalidGraphicsContext     );
		caseToStr(cudaErrorNvlinkUncorrectable        );
		caseToStr(cudaErrorStartupFailure             );
		caseToStr(cudaErrorApiFailureBase             );
	}

	return "Unknown error";
}

////////////////////////////////////////////////////////////////////////////////
// Check errors from the cuda runtime
void __cudaSafe(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
      std::cerr << "cudaSafe() Driver API error(" << err << "): " << cudaErrorToString(err)  << " from file " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////

cuda::cuda(size_t dev_id) {
  int deviceCount;
  char deviceName[100];

  cuInit(0);

  cuDeviceGetCount(&deviceCount);

  if(deviceCount <= dev_id) {LOG(FAIL, "Cuda bad device ID");}

  cuDeviceGet(&_cuDevice, dev_id);

  cuDeviceGetName(deviceName, 100, _cuDevice);
  LOG(DEBUG, "Using CUDA device: " + deviceName);

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, _cuDevice);

  LOG(DEBUG, "Major revision number:         "  + std::to_string(devProp.major              ));
  LOG(DEBUG, "Minor revision number:         "  + std::to_string(devProp.minor              ));
  LOG(DEBUG, "Total global memory:           "  + std::to_string(devProp.totalGlobalMem     ));
  LOG(DEBUG, "Total shared memory per block: "  + std::to_string(devProp.sharedMemPerBlock  ));
  LOG(DEBUG, "Total registers per block:     "  + std::to_string(devProp.regsPerBlock       ));
  LOG(DEBUG, "Warp size:                     "  + std::to_string(devProp.warpSize           ));
  LOG(DEBUG, "Maximum memory pitch:          "  + std::to_string(devProp.memPitch           ));
  LOG(DEBUG, "Maximum threads per block:     "  + std::to_string(devProp.maxThreadsPerBlock ));
  for (int i = 0; i < 3; ++i)
    LOG(DEBUG, "Maximum dimension " + std::to_string(i) + " of block:  " + std::to_string(devProp.maxThreadsDim[i]));
  for (int i = 0; i < 3; ++i)
    LOG(DEBUG, "Maximum dimension " + std::to_string(i) +  "of grid: " + std::to_string(devProp.maxGridSize[i]));
  LOG(DEBUG, "Clock rate:                    " + std::to_string( devProp.clockRate               ));
  LOG(DEBUG, "Total constant memory:         " + std::to_string( devProp.totalConstMem           ));
  LOG(DEBUG, "Texture alignment:             " + std::to_string( devProp.textureAlignment        ));
  LOG(DEBUG, "Concurrent copy and execution: " + (devProp.deviceOverlap ? "Yes" : "No"           ));
  LOG(DEBUG, "Number of multiprocessors:     " + std::to_string( devProp.multiProcessorCount     ));
  LOG(DEBUG, "Kernel execution timeout:      " + (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
}

////////////////////////////////////////////////////////////////////////////////
