#ifndef CUDF_RUST_MEMORY_RESOURCE_H
#define CUDF_RUST_MEMORY_RESOURCE_H

#include "error.h"
#include "stream.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an owning RMM device memory resource.
typedef struct cudf_memory_resource_handle_state* cudf_memory_resource_handle;

enum { CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS = 8 };
enum { CUDF_MEMORY_POOL_USAGE_RESERVED_SLOTS = 4 };
enum { CUDF_MEMORY_FAILURE_INFO_RESERVED_SLOTS = 4 };

enum {
  CUDF_MEMORY_FAILURE_INFO_HAS_POOL_USAGE        = 1u << 0,
  CUDF_MEMORY_FAILURE_INFO_HAS_ALLOCATION_DOMAIN = 1u << 1,
};

typedef enum {
  CUDF_MEMORY_FAILURE_ACTION_DECLINE        = 0,
  CUDF_MEMORY_FAILURE_ACTION_RETRY          = 1,
  CUDF_MEMORY_FAILURE_ACTION_TRIM_AND_RETRY = 2,
} cudf_memory_failure_action_t;

/// Reserved for the Rust unwind guard. Native callback adaptors must treat
/// this as a permanent callback-health failure, never as a user action.
enum { CUDF_MEMORY_FAILURE_ACTION_CALLBACK_PANICKED = (-2147483647 - 1) };

typedef enum {
  CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL          = 0,
  CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT = 1,
} cudf_memory_failure_source_t;

typedef enum {
  CUDF_MEMORY_FAILURE_DOMAIN_NONE                = 0,
  CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN = 3,
} cudf_memory_failure_domain_t;

/// Requested and aligned-byte counters for a memory resource.
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

/// Physical CUDA async-pool counters.
typedef struct {
  size_t struct_size;
  uint64_t reserved_current_bytes;
  uint64_t reserved_high_bytes;
  uint64_t used_current_bytes;
  uint64_t used_high_bytes;
  uint64_t reserved[CUDF_MEMORY_POOL_USAGE_RESERVED_SLOTS];
} cudf_memory_pool_usage_t;

/// Allocation-failure payload passed to observation callbacks.
typedef struct {
  size_t struct_size;
  int32_t device_ordinal;
  uint32_t flags;
  uint32_t callback_attempt;
  uint32_t trim_retry_count;
  uint64_t requested_bytes;
  uint64_t pool_reserved_current_bytes;
  uint64_t pool_reserved_high_bytes;
  uint64_t pool_used_current_bytes;
  uint64_t pool_used_high_bytes;
  uint32_t failure_source;
  uint32_t reserved_failure_info_padding;
  /// With `CUDF_MEMORY_FAILURE_INFO_HAS_ALLOCATION_DOMAIN`, `reserved[0]`
  /// stores a `cudf_memory_failure_domain_t`, then primary and member ids.
  uint64_t reserved[CUDF_MEMORY_FAILURE_INFO_RESERVED_SLOTS - 1];
} cudf_memory_failure_info_t;

/// Observation callback invoked after an upstream RMM allocation failure.
/// The return value is reserved and does not change allocation control flow;
/// callbacks must not allocate through the same resource.
typedef int32_t (*cudf_memory_failure_callback_t)(cudf_memory_failure_info_t const* info,
                                                  void* callback_arg);

cudf_status_t cudf_memory_resource_create_cuda(cudf_memory_resource_handle* out);
cudf_status_t cudf_memory_resource_create_pool(size_t initial_pool_size,
                                               size_t maximum_pool_size,
                                               cudf_memory_resource_handle* out);
cudf_status_t cudf_memory_resource_create_cuda_async(size_t initial_pool_size,
                                                     uint64_t release_threshold,
                                                     cudf_memory_resource_handle* out);
cudf_status_t cudf_memory_resource_create_failure_callback(cudf_memory_resource_handle upstream,
                                                           cudf_memory_failure_callback_t callback,
                                                           void* callback_arg,
                                                           cudf_memory_resource_handle* out);
cudf_status_t cudf_memory_resource_create_budget_tracking(cudf_memory_resource_handle upstream,
                                                          size_t allocation_limit,
                                                          cudf_memory_resource_handle* out);
cudf_status_t cudf_memory_resource_usage(cudf_memory_resource_handle handle,
                                         cudf_memory_resource_usage_t* out);

/// Return the hard allocation cap remaining after aligned live charges.
///
/// Budget-tracking and query-domain member resources ignore `cuda_stream`. A
/// device-manager resource resolves the active member bound to `cuda_stream`.
/// Query-domain resources report their parent domain's remaining cap.
/// Returns `CUDF_ERROR_UNSUPPORTED` when the resource has no hard cap.
cudf_status_t cudf_memory_resource_effective_device_memory(cudf_memory_resource_handle handle,
                                                            void* cuda_stream,
                                                            size_t* out);
cudf_status_t cudf_memory_resource_pool_usage(cudf_memory_resource_handle handle,
                                              cudf_memory_pool_usage_t* out);
cudf_status_t cudf_memory_resource_trim_to(cudf_memory_resource_handle handle, uint64_t keep_bytes);

/// Destroy on the supplied owning CUDA device. The call restores the previous
/// calling-thread device before returning.
cudf_status_t cudf_memory_resource_destroy(cudf_memory_resource_handle handle, int32_t device);
cudf_status_t cudf_memory_resource_raw(cudf_memory_resource_handle handle, void** out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_MEMORY_RESOURCE_H
