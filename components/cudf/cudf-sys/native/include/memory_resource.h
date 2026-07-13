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
typedef void* cudf_memory_resource_handle;

/// Opaque handle to the process-local memory manager for one CUDA device.
typedef void* cudf_memory_manager_handle;

/// Opaque handle to one stream-bound memory reservation.
typedef void* cudf_reservation_handle;

/// Opaque handle to one query-scoped reservation arena.
typedef void* cudf_reservation_arena_handle;

/// Opaque handle to one query-scoped memory domain.
typedef void* cudf_query_memory_domain_handle;

/// Callback invoked by a failure-callback memory resource after an upstream
/// allocation fails with an RMM out-of-memory error.
///
/// Legacy callback resources may trim idle async-pool pages and retry once
/// before invoking this callback. Use `cudf_memory_failure_callback_v2_t` when
/// the callback owner needs the failure payload and explicit trim action.
///
/// Return nonzero to retry the allocation, or zero to rethrow the allocation
/// failure. Implementations must be re-entrancy-safe: the callback runs inside
/// the allocating thread and must not allocate through the same memory resource.
typedef int32_t (*cudf_memory_failure_callback_t)(size_t bytes, void* callback_arg);

enum { CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS = 8 };
enum { CUDF_MEMORY_POOL_USAGE_RESERVED_SLOTS = 4 };
enum { CUDF_MEMORY_MANAGER_OPTIONS_RESERVED_SLOTS = 4 };
enum { CUDF_MEMORY_FAILURE_INFO_RESERVED_SLOTS = 4 };

enum {
  CUDF_MEMORY_FAILURE_INFO_HAS_POOL_USAGE = 1u << 0,
  CUDF_MEMORY_FAILURE_INFO_HAS_ALLOCATION_DOMAIN = 1u << 1,
};

typedef enum {
  CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC = 0,
  CUDF_MEMORY_MANAGER_BASE_POOL = 1,
} cudf_memory_manager_base_kind_t;

typedef enum {
  CUDF_MEMORY_FAILURE_ACTION_DECLINE = 0,
  CUDF_MEMORY_FAILURE_ACTION_RETRY = 1,
  CUDF_MEMORY_FAILURE_ACTION_TRIM_AND_RETRY = 2,
} cudf_memory_failure_action_t;

typedef enum {
  CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL = 0,
  CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT = 1,
} cudf_memory_failure_source_t;

typedef enum {
  CUDF_MEMORY_FAILURE_DOMAIN_NONE = 0,
  CUDF_MEMORY_FAILURE_DOMAIN_STREAM_RESERVATION = 1,
  CUDF_MEMORY_FAILURE_DOMAIN_RESERVATION_ARENA = 2,
  CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN = 3,
} cudf_memory_failure_domain_t;

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

/// Physical CUDA memory-pool counters for an async-pool-backed memory resource.
typedef struct {
  size_t struct_size;
  uint64_t reserved_current_bytes;
  uint64_t reserved_high_bytes;
  uint64_t used_current_bytes;
  uint64_t used_high_bytes;
  uint64_t reserved[CUDF_MEMORY_POOL_USAGE_RESERVED_SLOTS];
} cudf_memory_pool_usage_t;

/// Allocation-failure payload passed to reclaim-capable failure callbacks.
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
  /// `cudf_memory_failure_source_t` value identifying whether retry/trim can
  /// address the failure.
  uint32_t failure_source;
  uint32_t reserved_failure_info_padding;
  /// When `CUDF_MEMORY_FAILURE_INFO_HAS_ALLOCATION_DOMAIN` is set:
  /// reserved[0] is a `cudf_memory_failure_domain_t`, reserved[1] is the
  /// primary domain id, and reserved[2] is the member/reservation id when
  /// applicable. This overlay intentionally consumes every current reserved
  /// slot; future failure-info extensions must grow this struct or define a
  /// mutually exclusive overlay under a new flag.
  uint64_t reserved[CUDF_MEMORY_FAILURE_INFO_RESERVED_SLOTS - 1];
} cudf_memory_failure_info_t;

/// Reclaim-capable callback invoked by a failure-callback memory resource.
///
/// Return a `cudf_memory_failure_action_t` value. For physical upstream
/// allocation failures, `TRIM_AND_RETRY` asks the native adaptor to trim the
/// async pool to zero idle bytes, then retry the allocation if trimming
/// succeeds. Reservation-limit failures are reported to the callback but never
/// trim the physical pool.
typedef int32_t (*cudf_memory_failure_callback_v2_t)(cudf_memory_failure_info_t const* info,
                                                     void* callback_arg);

/// Device memory manager physical-base options.
///
/// Default options use `CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC`, no initial pool
/// priming, and a release threshold of `UINT64_MAX` so the pool stays warm
/// until callers explicitly trim it.
typedef struct {
  size_t struct_size;
  cudf_memory_manager_base_kind_t base_kind;
  uint32_t flags;
  uint64_t initial_pool_size_bytes;
  uint64_t maximum_pool_size_bytes;
  uint64_t release_threshold_bytes;
  uint64_t reserved[CUDF_MEMORY_MANAGER_OPTIONS_RESERVED_SLOTS];
} cudf_memory_manager_options_t;

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

/// Create a cudaMallocAsync-backed memory resource.
///
/// `initial_pool_size` primes the pool when nonzero. `release_threshold`
/// controls the CUDA pool release threshold; pass `UINT64_MAX` to retain idle
/// memory across synchronizations until an explicit trim.
cudf_status_t cudf_memory_resource_create_cuda_async(size_t initial_pool_size,
                                                     uint64_t release_threshold,
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

/// Create a failure-callback memory resource that wraps an existing memory resource.
///
/// The returned resource calls `callback(bytes, callback_arg)` when the
/// upstream resource throws an RMM out-of-memory error. A nonzero callback
/// result retries the allocation; zero preserves the original failure. The
/// callback and its argument must remain valid until the returned resource is
/// destroyed. Existing upstream allocations are not counted. The returned
/// handle retains the upstream resource lifetime and does not change the
/// allocator used by unrelated `cudf-sys` APIs.
cudf_status_t cudf_memory_resource_create_failure_callback(
  cudf_memory_resource_handle upstream,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudf_memory_resource_handle* out);

/// Create a reclaim-capable failure-callback memory resource around `upstream`.
cudf_status_t cudf_memory_resource_create_failure_callback_v2(
  cudf_memory_resource_handle upstream,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
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

/// Read physical async-pool counters.
///
/// Returns `CUDF_ERROR_INVALID_OPERATION` if `handle` is not backed by a CUDA
/// async pool.
cudf_status_t cudf_memory_resource_pool_usage(cudf_memory_resource_handle handle,
                                              cudf_memory_pool_usage_t* out);

/// Explicitly trim idle bytes from a CUDA async pool.
///
/// `keep_bytes` is passed to `cudaMemPoolTrimTo`. Returns
/// `CUDF_ERROR_INVALID_OPERATION` if `handle` is not backed by a CUDA async
/// pool.
cudf_status_t cudf_memory_resource_trim_to(cudf_memory_resource_handle handle,
                                           uint64_t keep_bytes);

/// Resolve the process-local reservation-aware memory manager for `device`.
///
/// The manager owns one device-level reservation-aware adaptor backed by one
/// shared physical pool. The adaptor is installed as the CUDA device's current
/// RMM resource when the manager is first created; subsequent calls return the
/// same manager without replacing the current resource.
cudf_status_t cudf_memory_manager_for_device(int32_t device, cudf_memory_manager_handle* out);

/// Resolve the process-local reservation-aware memory manager for `device`
/// with explicit physical-base options.
///
/// The options only affect first creation for a device. Later calls return the
/// already-created manager.
cudf_status_t cudf_memory_manager_for_device_with_options(
  int32_t device,
  cudf_memory_manager_options_t const* options,
  cudf_memory_manager_handle* out);

/// Resolve the process-local reservation-aware memory manager for `device`,
/// installing a failure callback around the device-global current resource if
/// this call creates the manager.
///
/// The callback is installed outside the reservation-aware adaptor, so it
/// observes both physical upstream OOMs and reservation-limit failures. If the
/// manager already exists, this function succeeds only when that manager was
/// created with a failure callback; it does not replace an existing current
/// device resource or callback. `callback_arg` is stored by the process-global
/// device manager and must remain valid for the process/device-manager
/// lifetime.
cudf_status_t cudf_memory_manager_for_device_with_failure_callback(
  int32_t device,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudf_memory_manager_handle* out);

/// Resolve the process-local reservation-aware memory manager for `device`,
/// installing a reclaim-capable process-lifetime failure callback if this call
/// creates the manager.
cudf_status_t cudf_memory_manager_for_device_with_failure_callback_v2(
  int32_t device,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
  cudf_memory_manager_handle* out);

/// Resolve the process-local reservation-aware memory manager for `device`
/// with explicit physical-base options and a process-lifetime failure callback.
cudf_status_t cudf_memory_manager_for_device_with_failure_callback_and_options(
  int32_t device,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudf_memory_manager_options_t const* options,
  cudf_memory_manager_handle* out);

/// Resolve the process-local reservation-aware memory manager for `device`
/// with options and a reclaim-capable process-lifetime failure callback.
cudf_status_t cudf_memory_manager_for_device_with_failure_callback_v2_and_options(
  int32_t device,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
  cudf_memory_manager_options_t const* options,
  cudf_memory_manager_handle* out);

/// Query whether a process-local memory manager already exists for `device`
/// and whether it was created with a failure callback.
///
/// `*exists` is set to nonzero when a manager exists. `*failure_callback_installed`
/// is set to nonzero only when that existing manager was callback-installed.
cudf_status_t cudf_memory_manager_failure_callback_state(
  int32_t device,
  int32_t* exists,
  int32_t* failure_callback_installed);

/// Create a memory-resource handle alias for a device memory manager.
///
/// The returned handle points at the manager's long-lived device-level adaptor
/// and can be passed through `cudf_exec_options_t::mr`. Destroying this handle
/// releases only the handle alias; the manager and its current-resource
/// installation remain alive for the process.
cudf_status_t cudf_memory_manager_resource(cudf_memory_manager_handle manager,
                                           cudf_memory_resource_handle* out);

/// Read physical async-pool counters for a device memory manager.
cudf_status_t cudf_memory_manager_pool_usage(cudf_memory_manager_handle manager,
                                             cudf_memory_pool_usage_t* out);

/// Explicitly trim idle bytes from a device memory manager's CUDA async pool.
cudf_status_t cudf_memory_manager_trim_to(cudf_memory_manager_handle manager,
                                          uint64_t keep_bytes);

/// Attach a hard live-byte reservation to `stream` for this manager.
///
/// Allocations routed through the manager's resource are charged to the
/// reservation currently attached to the allocation stream. Deallocation
/// releases the allocation identity that created the pointer, even after the
/// reservation has detached from the stream.
cudf_status_t cudf_memory_manager_attach_reservation(cudf_memory_manager_handle manager,
                                                     cudf_stream_handle stream,
                                                     uint64_t reservation_id,
                                                     size_t limit_bytes,
                                                     cudf_reservation_handle* out);

/// Create a query-scoped reservation arena for this manager.
///
/// Member reservations attached to the arena are still stream-bound, but every
/// allocation is also charged to this parent arena before the member is charged.
/// The arena limit is a hard live-byte cap shared by all members.
cudf_status_t cudf_memory_manager_create_reservation_arena(
  cudf_memory_manager_handle manager,
  uint64_t arena_id,
  size_t limit_bytes,
  cudf_reservation_arena_handle* out);

/// Attach a stream-bound reservation as a member of an arena.
cudf_status_t cudf_memory_manager_attach_reservation_to_arena(
  cudf_memory_manager_handle manager,
  cudf_stream_handle stream,
  cudf_reservation_arena_handle arena,
  uint64_t reservation_id,
  size_t limit_bytes,
  cudf_reservation_handle* out);

/// Read usage counters for a live reservation handle.
cudf_status_t cudf_reservation_usage(cudf_reservation_handle reservation,
                                     cudf_memory_resource_usage_t* out);

/// Resize the live-byte limit for an attached reservation.
///
/// Shrinking below current live usage is allowed; future allocations fail until
/// enough memory is released or the reservation is grown again.
cudf_status_t cudf_reservation_resize(cudf_reservation_handle reservation,
                                      size_t limit_bytes);

/// Detach a reservation from its stream and destroy the reservation handle.
///
/// Retained allocations remain associated with the detached reservation for
/// accounting until those pointers are deallocated.
cudf_status_t cudf_reservation_detach(cudf_reservation_handle reservation);

/// Read usage counters for a live reservation arena handle.
cudf_status_t cudf_reservation_arena_usage(cudf_reservation_arena_handle arena,
                                           cudf_memory_resource_usage_t* out);

/// Resize the live-byte limit for a reservation arena.
///
/// Shrinking below current live usage is allowed; future member allocations fail
/// until enough memory is released or the arena is grown again.
cudf_status_t cudf_reservation_arena_resize(cudf_reservation_arena_handle arena,
                                            size_t limit_bytes);

/// Destroy an arena handle.
///
/// Retained allocations remain associated with the arena for accounting until
/// those pointers are deallocated.
cudf_status_t cudf_reservation_arena_destroy(cudf_reservation_arena_handle arena);

/// Create an opt-in query memory domain for this manager.
///
/// The domain owns a CUDA async memory pool capped by `ceiling_bytes` and a
/// resizable live-byte grant gate initially set to `grant_bytes`. `ceiling_bytes`
/// must be nonzero and `grant_bytes <= ceiling_bytes`. The manager's public
/// device-level memory resource remains the resource passed through execution
/// options; streams attached to the domain are routed internally.
cudf_status_t cudf_memory_manager_create_query_memory_domain(
  cudf_memory_manager_handle manager,
  uint64_t domain_id,
  size_t grant_bytes,
  size_t ceiling_bytes,
  uint64_t release_threshold_bytes,
  cudf_query_memory_domain_handle* out);

/// Attach a stream to a query memory domain.
///
/// Allocations routed through the manager's public memory resource on this
/// stream are served by the domain's pool and charged to the domain grant.
cudf_status_t cudf_query_memory_domain_attach_stream(cudf_query_memory_domain_handle domain,
                                                     cudf_stream_handle stream,
                                                     uint64_t reservation_id,
                                                     cudf_reservation_handle* out);

/// Read usage counters for a query memory domain.
cudf_status_t cudf_query_memory_domain_usage(cudf_query_memory_domain_handle domain,
                                             cudf_memory_resource_usage_t* out);

/// Read physical CUDA async-pool counters for a query memory domain.
cudf_status_t cudf_query_memory_domain_pool_usage(cudf_query_memory_domain_handle domain,
                                                  cudf_memory_pool_usage_t* out);

/// Resize the domain's live-byte grant.
///
/// Shrinking below current live usage is allowed; future allocations fail until
/// enough memory is released or the grant is grown. Growing above the immutable
/// driver pool ceiling is rejected.
cudf_status_t cudf_query_memory_domain_resize_grant(cudf_query_memory_domain_handle domain,
                                                    size_t grant_bytes);

/// Explicitly trim idle bytes from the domain's CUDA async pool.
cudf_status_t cudf_query_memory_domain_trim(cudf_query_memory_domain_handle domain,
                                            uint64_t keep_bytes);

/// Destroy a query memory domain handle.
///
/// The domain stops accepting new stream attachments. Retained allocations keep
/// the native domain alive until they are released.
cudf_status_t cudf_query_memory_domain_destroy(cudf_query_memory_domain_handle domain);

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

/// Read the number of private query-domain pools retained for one domain id.
///
/// Test-only hook for lifecycle/routing assertions.
cudf_status_t cudf_test_memory_manager_query_domain_pool_count_for_id(
  cudf_memory_manager_handle manager,
  uint64_t domain_id,
  size_t* out);

/// Arm a one-shot test pause after query-domain routing has acquired its
/// in-flight lifetime guard and before the private-pool allocation begins.
///
/// Returns nonzero when debug test hooks are compiled in. Release builds
/// return zero and do not add a hook check to the allocation hot path.
int32_t cudf_test_query_domain_route_pause_arm(uint64_t domain_id);

/// Return nonzero after the armed query-domain route pause is reached.
int32_t cudf_test_query_domain_route_pause_reached(void);

/// Release an armed or reached query-domain route pause.
void cudf_test_query_domain_route_pause_release(void);

/// Create a statistics-wrapped memory resource for allocator-sensitive tests.
///
/// Test-only hook. The returned handle owns an RMM statistics adaptor and can
/// be inspected with `cudf_test_memory_resource_allocation_count`.
cudf_status_t cudf_test_memory_resource_create_stats(cudf_memory_resource_handle* out);

/// Create a test-only resource that fails the first allocation, then delegates
/// all later allocations to `upstream`.
cudf_status_t cudf_test_memory_resource_create_fail_first_allocation(
  cudf_memory_resource_handle upstream,
  cudf_memory_resource_handle* out);

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
