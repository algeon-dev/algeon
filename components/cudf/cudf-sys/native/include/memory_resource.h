#ifndef CUDF_RUST_MEMORY_RESOURCE_H
#define CUDF_RUST_MEMORY_RESOURCE_H

#include "error.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an owning RMM device memory resource.
typedef void* cudf_memory_resource_handle;

/// Create a CUDA (default) memory resource.
///
/// This creates a standalone owned RMM resource object. The rest of the
/// `cudf-sys` compute surface does not currently accept or install memory
/// resource handles, and continues to allocate via
/// `cudf::get_current_device_resource_ref()`.
cudf_status_t cudf_memory_resource_create_cuda(cudf_memory_resource_handle* out);

/// Create a pool memory resource.
///
/// `initial_pool_size` and `maximum_pool_size` are in bytes.
/// Pass 0 for `maximum_pool_size` to allow unbounded growth.
///
/// This does not change the allocator used by other `cudf-sys` APIs.
cudf_status_t cudf_memory_resource_create_pool(size_t initial_pool_size,
                                                size_t maximum_pool_size,
                                                cudf_memory_resource_handle* out);

/// Destroy a memory resource handle.
void cudf_memory_resource_destroy(cudf_memory_resource_handle handle);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_MEMORY_RESOURCE_H
