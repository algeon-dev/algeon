#include "cudf_rust/device.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <cuda_profiler_api.h>

#include <limits>
#include <string>

namespace {
bool cudf_device_attribute_to_cuda_attribute(int32_t attr, cudaDeviceAttr* out)
{
  switch (attr) {
    case CUDF_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
      *out = cudaDevAttrMultiProcessorCount;
      return true;
    case CUDF_DEVICE_ATTRIBUTE_MANAGED_MEMORY: *out = cudaDevAttrManagedMemory; return true;
    case CUDF_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
      *out = cudaDevAttrConcurrentManagedAccess;
      return true;
    default: return false;
  }
}
}  // namespace

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

cudf_status_t cudf_cuda_profiler_start(void){CUDF_RUST_TRY{cudaError_t err = cudaProfilerStart();
if (err != cudaSuccess) {
  auto msg = std::string("cudaProfilerStart failed: ") + cudaGetErrorName(err) + " (code " +
             std::to_string(static_cast<int>(err)) + "): " + cudaGetErrorString(err);
  set_error(msg);
  return CUDF_ERROR_CUDA;
}
return CUDF_SUCCESS;
}
CUDF_RUST_CATCH
}

cudf_status_t cudf_cuda_profiler_stop(void){CUDF_RUST_TRY{cudaError_t err = cudaProfilerStop();
if (err != cudaSuccess) {
  auto msg = std::string("cudaProfilerStop failed: ") + cudaGetErrorName(err) + " (code " +
             std::to_string(static_cast<int>(err)) + "): " + cudaGetErrorString(err);
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

cudf_status_t cudf_cuda_device_pci_bus_id(int32_t device, char* out, size_t out_len)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY
  {
    if (out_len == 0) {
      set_error("out_len must be positive");
      return CUDF_ERROR_INVALID_ARGUMENT;
    }
    out[0] = '\0';
    if (out_len > static_cast<size_t>(std::numeric_limits<int>::max())) {
      set_error("out_len exceeds CUDA API limit");
      return CUDF_ERROR_INVALID_ARGUMENT;
    }

    cudf_device_guard guard;
    cudf_status_t const guard_status = guard.activate(device);
    if (guard_status != CUDF_SUCCESS) { return guard_status; }

    cudaError_t err = cudaDeviceGetPCIBusId(out, static_cast<int>(out_len), device);
    if (err != cudaSuccess) {
      auto msg = std::string("cudaDeviceGetPCIBusId failed: ") + cudaGetErrorString(err);
      set_error(msg);
      return CUDF_ERROR_CUDA;
    }

    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_device_memory_info(int32_t device, size_t* free_bytes, size_t* total_bytes)
{
  CUDF_RUST_NULL_CHECK(free_bytes, "free_bytes is null");
  CUDF_RUST_NULL_CHECK(total_bytes, "total_bytes is null");
  *free_bytes  = 0;
  *total_bytes = 0;
  CUDF_RUST_TRY
  {
    cudf_device_guard guard;
    cudf_status_t const guard_status = guard.activate(device);
    if (guard_status != CUDF_SUCCESS) { return guard_status; }

    auto const [free, total] = rmm::available_device_memory();
    *free_bytes              = free;
    *total_bytes             = total;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_device_attribute(int32_t device, int32_t attr, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_TRY
  {
    cudf_device_guard guard;
    cudf_status_t const guard_status = guard.activate(device);
    if (guard_status != CUDF_SUCCESS) { return guard_status; }

    cudaDeviceAttr cuda_attr{};
    if (!cudf_device_attribute_to_cuda_attribute(attr, &cuda_attr)) {
      set_error("unsupported cudf device attribute");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    int32_t value   = 0;
    cudaError_t err = cudaDeviceGetAttribute(&value, cuda_attr, device);
    if (err != cudaSuccess) {
      auto msg = std::string("cudaDeviceGetAttribute failed: ") + cudaGetErrorString(err);
      set_error(msg);
      return CUDF_ERROR_CUDA;
    }

    *out = value;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
