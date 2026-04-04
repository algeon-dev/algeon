#ifndef CUDF_RUST_MEMORY_RESOURCE_H
#define CUDF_RUST_MEMORY_RESOURCE_H

#include "error.h"
#include <stddef.h>
#include <stdint.h>

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

/// Borrow the underlying `device_memory_resource*` from an owning memory
/// resource handle.
///
/// This does not transfer ownership. `*out` is valid only while `handle`
/// remains alive.
cudf_status_t cudf_memory_resource_raw(
    cudf_memory_resource_handle handle, void** out);

/// Create a statistics-wrapped memory resource for allocator-sensitive tests.
///
/// Test-only hook. The returned handle owns an RMM statistics adaptor and can
/// be inspected with `cudf_test_memory_resource_allocation_count`.
cudf_status_t cudf_test_memory_resource_create_stats(
    cudf_memory_resource_handle* out);

/// Read the cumulative allocation count from a statistics memory resource.
///
/// Test-only hook. Fails if `handle` does not refer to a statistics-enabled
/// memory resource created by `cudf_test_memory_resource_create_stats`.
cudf_status_t cudf_test_memory_resource_allocation_count(
    cudf_memory_resource_handle handle, int64_t* out);

/// Resolve a memory-resource handle to its underlying `device_memory_resource*`.
///
/// Test-only hook for debug assertions that need the exact pointer recorded by
/// `cudf_resolve_exec_options`.
cudf_status_t cudf_test_memory_resource_raw(
    cudf_memory_resource_handle handle, void** out);

/// Install the memory resource owned by `handle` as the process-global current
/// device resource.
///
/// Test-only hook for allocator-sensitive integration tests.
cudf_status_t cudf_test_memory_resource_set_current(
    cudf_memory_resource_handle handle);

/// Install `raw_mr` as the process-global current device resource.
///
/// Test-only hook intended for restoring a previously observed resource from
/// `cudf_test_current_device_resource_raw`.
cudf_status_t cudf_test_memory_resource_set_current_raw(void* raw_mr);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_MEMORY_RESOURCE_H
