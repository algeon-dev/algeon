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

enum { CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS = 8 };

/// Usage counters exposed by a production budget-tracking memory resource.
///
/// `current_bytes` / `peak_bytes` are requested live bytes as seen by the
/// resource API. `current_aligned_bytes` / `peak_aligned_bytes` are the aligned
/// byte counters used for hard-limit enforcement.
typedef struct {
  size_t struct_size;
  uint64_t allocation_limit_bytes;
  uint64_t current_bytes;
  uint64_t peak_bytes;
  uint64_t total_bytes;
  uint64_t current_aligned_bytes;
  uint64_t peak_aligned_bytes;
  uint64_t total_allocations;
  uint64_t current_allocations;
  uint64_t peak_allocations;
  uint64_t failed_allocation_count;
  uint64_t largest_failed_allocation_bytes;
  uint64_t reserved[CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS];
} cudf_memory_resource_usage_t;

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

/// Create a limiting memory resource that wraps an existing memory resource.
///
/// `allocation_limit` is the maximum number of live bytes this adaptor will
/// allow through itself. Existing upstream allocations are not counted. The
/// returned handle retains the upstream resource lifetime and does not change
/// the allocator used by unrelated `cudf-sys` APIs.
cudf_status_t cudf_memory_resource_create_limiting(cudf_memory_resource_handle upstream,
                                                   size_t allocation_limit,
                                                   cudf_memory_resource_handle* out);

/// Create a budget-tracking memory resource that wraps an existing memory resource.
///
/// The returned resource enforces a hard aligned live-byte limit and records
/// operation-scoped current/peak allocation counters. Existing upstream
/// allocations are not counted. The returned handle retains the upstream
/// resource lifetime and does not change the allocator used by unrelated
/// `cudf-sys` APIs.
cudf_status_t cudf_memory_resource_create_budget_tracking(cudf_memory_resource_handle upstream,
                                                          size_t allocation_limit,
                                                          cudf_memory_resource_handle* out);

/// Read usage counters from a budget-tracking memory resource.
///
/// The returned counters are diagnostic snapshots loaded independently; this
/// API does not provide a multi-field atomic live-polling snapshot. Operation
/// reports should query usage after the budgeted call returns.
///
/// Returns `CUDF_ERROR_INVALID_OPERATION` if `handle` does not refer to a
/// budget-tracking resource created by
/// `cudf_memory_resource_create_budget_tracking`.
cudf_status_t cudf_memory_resource_usage(cudf_memory_resource_handle handle,
                                         cudf_memory_resource_usage_t* out);

/// Destroy a memory resource handle on the CUDA device that owns it.
///
/// The shim temporarily switches the calling thread to `device`, destroys the
/// resource, then restores the previous device before returning.
///
/// Returns `CUDF_ERROR_INVALID_OPERATION` if `handle` is currently installed as
/// RMM's process-global current device resource. Callers must first install a
/// different current resource before destroying this handle.
cudf_status_t cudf_memory_resource_destroy(cudf_memory_resource_handle handle, int32_t device);

/// Borrow the underlying `device_memory_resource*` from an owning memory
/// resource handle.
///
/// This does not transfer ownership. `*out` is valid only while `handle`
/// remains alive.
cudf_status_t cudf_memory_resource_raw(cudf_memory_resource_handle handle, void** out);

/// Create a statistics-wrapped memory resource for allocator-sensitive tests.
///
/// Test-only hook. The returned handle owns an RMM statistics adaptor and can
/// be inspected with `cudf_test_memory_resource_allocation_count`.
cudf_status_t cudf_test_memory_resource_create_stats(cudf_memory_resource_handle* out);

/// Read the cumulative allocation count from a statistics memory resource.
///
/// Test-only hook. Fails if `handle` does not refer to a statistics-enabled
/// memory resource created by `cudf_test_memory_resource_create_stats`.
cudf_status_t cudf_test_memory_resource_allocation_count(cudf_memory_resource_handle handle,
                                                         int64_t* out);

/// Read the current live allocation count from a statistics memory resource.
///
/// Test-only hook. Fails if `handle` does not refer to a statistics-enabled
/// memory resource created by `cudf_test_memory_resource_create_stats`.
cudf_status_t cudf_test_memory_resource_active_allocation_count(cudf_memory_resource_handle handle,
                                                                int64_t* out);

/// Resolve a memory-resource handle to its underlying `device_memory_resource*`.
///
/// Test-only hook for debug assertions that need the exact pointer recorded by
/// `cudf_resolve_exec_options`.
cudf_status_t cudf_test_memory_resource_raw(cudf_memory_resource_handle handle, void** out);

/// Install the memory resource owned by `handle` as the process-global current
/// device resource.
///
/// Test-only hook for allocator-sensitive integration tests.
cudf_status_t cudf_test_memory_resource_set_current(cudf_memory_resource_handle handle);

/// Install `raw_mr` as the process-global current device resource.
///
/// Test-only hook intended for restoring a previously observed resource from
/// `cudf_test_current_device_resource_raw`.
cudf_status_t cudf_test_memory_resource_set_current_raw(void* raw_mr);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_MEMORY_RESOURCE_H
