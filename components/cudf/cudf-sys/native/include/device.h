#ifndef CUDF_RUST_DEVICE_H
#define CUDF_RUST_DEVICE_H

#include "error.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Query the CUDA device ordinal for the calling thread.
cudf_status_t cudf_cuda_get_device(int32_t* device);

/// Set the CUDA device ordinal for the calling thread.
cudf_status_t cudf_cuda_set_device(int32_t device);

/// Query the number of CUDA-capable devices visible to the process.
cudf_status_t cudf_cuda_get_device_count(int32_t* count);

/// Resolve the CUDA device ordinal that owns a device pointer.
cudf_status_t cudf_cuda_pointer_get_device(const void* ptr, int32_t* device);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_DEVICE_H
