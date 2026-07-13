#ifndef CUDF_RUST_DEVICE_H
#define CUDF_RUST_DEVICE_H

#include "error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Query the CUDA device ordinal for the calling thread.
cudf_status_t cudf_cuda_get_device(int32_t *device);

/// Set the CUDA device ordinal for the calling thread.
cudf_status_t cudf_cuda_set_device(int32_t device);

/// Start profiler collection for the current process.
cudf_status_t cudf_cuda_profiler_start(void);

/// Stop profiler collection for the current process.
cudf_status_t cudf_cuda_profiler_stop(void);

/// Query the number of CUDA-capable devices visible to the process.
cudf_status_t cudf_cuda_get_device_count(int32_t *count);

/// Resolve the CUDA device ordinal that owns a device pointer.
cudf_status_t cudf_cuda_pointer_get_device(const void *ptr, int32_t *device);

/// Query the CUDA PCI bus id for a CUDA device.
cudf_status_t cudf_cuda_device_pci_bus_id(int32_t device, char *out,
                                          size_t out_len);

/// Device attributes exposed by the cudf-rs CUDA inventory shim.
typedef enum {
  CUDF_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 1,
  CUDF_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 2,
  CUDF_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 3,
} cudf_device_attribute_t;

/// Query free and total memory for a CUDA device.
cudf_status_t cudf_device_memory_info(int32_t device, size_t *free_bytes,
                                      size_t *total_bytes);

/// Query a CUDA device attribute selected by cudf_device_attribute_t.
cudf_status_t cudf_device_attribute(int32_t device, int32_t attr, int32_t *out);

#ifdef __cplusplus
}
#endif

#endif // CUDF_RUST_DEVICE_H
