#include "device.h"

#include "shim_common.h"

#include <cuda_runtime_api.h>

#include <string>

extern "C" {

cudf_status_t cudf_cuda_get_device(int32_t* device)
{
  CUDF_RUST_NULL_CHECK(device, "device is null");
  *device = 0;
  CUDF_RUST_TRY
  {
    cudaError_t err = cudaGetDevice(device);
    if (err != cudaSuccess) {
      auto msg = std::string("cudaGetDevice failed: ") + cudaGetErrorString(err);
      set_error(msg);
      return CUDF_ERROR_CUDA;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_cuda_set_device(int32_t device){
  CUDF_RUST_TRY{cudaError_t err = cudaSetDevice(device);
if (err != cudaSuccess) {
  auto msg = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
  set_error(msg);
  return CUDF_ERROR_CUDA;
}
return CUDF_SUCCESS;
}
CUDF_RUST_CATCH
}

cudf_status_t cudf_cuda_get_device_count(int32_t* count)
{
  CUDF_RUST_NULL_CHECK(count, "count is null");
  *count = 0;
  CUDF_RUST_TRY
  {
    cudaError_t err = cudaGetDeviceCount(count);
    if (err != cudaSuccess) {
      auto msg = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err);
      set_error(msg);
      return CUDF_ERROR_CUDA;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_cuda_pointer_get_device(void const* ptr, int32_t* device)
{
  CUDF_RUST_NULL_CHECK(ptr, "ptr is null");
  CUDF_RUST_NULL_CHECK(device, "device is null");
  *device = 0;
  CUDF_RUST_TRY
  {
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
      auto msg = std::string("cudaPointerGetAttributes failed: ") + cudaGetErrorString(err);
      set_error(msg);
      return CUDF_ERROR_CUDA;
    }
#if CUDART_VERSION >= 10000
    if (attrs.type != cudaMemoryTypeDevice) {
      set_error("pointer is not CUDA device memory");
      return CUDF_ERROR_INVALID_OPERATION;
    }
#else
    if (attrs.memoryType != cudaMemoryTypeDevice) {
      set_error("pointer is not CUDA device memory");
      return CUDF_ERROR_INVALID_OPERATION;
    }
#endif
    *device = attrs.device;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
