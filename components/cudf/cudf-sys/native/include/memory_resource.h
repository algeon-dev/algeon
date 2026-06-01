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
/// This creates an owned RMM resource object that can be attached to
/// `cudf_exec_options_t` for `_ex` APIs. It does not change the process-global
/// allocator used by unrelated operations.
cudf_status_t cudf_memory_resource_create_cuda(cudf_memory_resource_handle* out);

/// Create a pool memory resource.
///
/// `initial_pool_size` and `maximum_pool_size` are in bytes.
/// Pass 0 for `maximum_pool_size` to allow unbounded growth.
///
/// This does not change the allocator used by unrelated `cudf-sys` APIs.
cudf_status_t cudf_memory_resource_create_pool(size_t initial_pool_size,
                                                size_t maximum_pool_size,
                                                cudf_memory_resource_handle* out);

/// Destroy a memory resource handle.
void cudf_memory_resource_destroy(cudf_memory_resource_handle handle);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_MEMORY_RESOURCE_H
