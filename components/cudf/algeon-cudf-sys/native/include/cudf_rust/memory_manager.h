#ifndef CUDF_RUST_MEMORY_MANAGER_H
#define CUDF_RUST_MEMORY_MANAGER_H

#include "allocator_event.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to the process-local memory manager for one CUDA device.
typedef struct cudf_memory_manager_handle_state* cudf_memory_manager_handle;

/// Opaque lease that routes current-RMM allocations for one CUDA stream to an
/// explicit memory resource without replacing the device's current resource.
typedef struct cudf_current_resource_route_handle_state* cudf_current_resource_route_handle;

// allocation_limit_bytes consumes the first reserved slot from the original
// version of this ABI. Keep the complete options record at 72 bytes so callers
// compiled against the original four-slot layout remain binary-compatible.
enum { CUDF_MEMORY_MANAGER_OPTIONS_RESERVED_SLOTS = 3 };

typedef enum {
  CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC = 0,
  CUDF_MEMORY_MANAGER_BASE_POOL       = 1,
} cudf_memory_manager_base_kind_t;

/// Physical-base configuration fixed when a device manager is first created.
typedef struct {
  size_t struct_size;
  cudf_memory_manager_base_kind_t base_kind;
  uint32_t flags;
  uint64_t initial_pool_size_bytes;
  uint64_t maximum_pool_size_bytes;
  uint64_t release_threshold_bytes;
  /// Optional manager-wide live-allocation limit. Zero means unbounded.
  uint64_t allocation_limit_bytes;
  uint64_t reserved[CUDF_MEMORY_MANAGER_OPTIONS_RESERVED_SLOTS];
} cudf_memory_manager_options_t;

typedef struct {
  uint8_t owner_fingerprint[16];
  uint32_t callback_version;
  uint32_t reserved_padding;
  uint8_t partition_fingerprint[16];
} cudf_memory_manager_installation_identity_t;

typedef enum {
  CUDF_MEMORY_MANAGER_INSTALLATION_UNINSTALLED       = 0,
  CUDF_MEMORY_MANAGER_INSTALLATION_INCOMPLETE        = 1,
  CUDF_MEMORY_MANAGER_INSTALLATION_OWNED             = 2,
  CUDF_MEMORY_MANAGER_INSTALLATION_UNIDENTIFIED      = 3,
  CUDF_MEMORY_MANAGER_INSTALLATION_CALLBACK_PANICKED = 4,
} cudf_memory_manager_installation_status_t;

typedef enum {
  CUDF_MEMORY_MANAGER_INSTALLATION_INSTALLABLE          = 0,
  CUDF_MEMORY_MANAGER_INSTALLATION_RETRYABLE            = 1,
  CUDF_MEMORY_MANAGER_INSTALLATION_REUSABLE             = 2,
  CUDF_MEMORY_MANAGER_INSTALLATION_PERMANENTLY_UNUSABLE = 3,
} cudf_memory_manager_installation_retry_t;

/// Closed reason carried by every authoritative install/inspect record.
typedef enum {
  CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_NONE                            = 0,
  CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CURRENT_RESOURCE_INSTALL_FAILED = 1,
  CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CURRENT_RESOURCE_REPLACED       = 2,
  CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_FAILURE_CALLBACK_PANICKED       = 3,
  CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_FAILURE_CALLBACK_MISSING        = 4,
  CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_IDENTITY_MISSING                = 5,
  CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_OPTIONS_MISMATCH                = 6,
  CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_IDENTITY_MISMATCH               = 7,
  CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CALLBACK_BINDING_MISMATCH       = 8,
} cudf_memory_manager_installation_cause_t;

enum {
  CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_EXISTS                     = 1u << 0,
  CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_FAILURE_CALLBACK_INSTALLED = 1u << 1,
  CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_CURRENT_RESOURCE_INSTALLED = 1u << 2,
  CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_CALLBACK_PANICKED          = 1u << 3,
  CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_HAS_IDENTITY               = 1u << 4,
  /// The manager was created by this install call, even if its current-resource
  /// switch remained incomplete.
  CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_CREATED_BY_CALL = 1u << 5,
};

/// Atomic process installation record for one device manager.
typedef struct {
  size_t struct_size;
  int32_t device_ordinal;
  uint32_t flags;
  cudf_memory_manager_installation_status_t status;
  cudf_memory_manager_installation_retry_t retry;
  cudf_memory_manager_installation_cause_t cause;
  uint32_t reserved_padding;
  uint64_t installation_generation;
  cudf_memory_manager_options_t effective_options;
  cudf_memory_manager_installation_identity_t identity;
  uint64_t reserved[3];
} cudf_memory_manager_installation_record_t;

cudf_status_t cudf_memory_manager_for_device(int32_t device, cudf_memory_manager_handle* out);
/// Explicitly install or reuse a device manager with immutable physical options.
/// `identity` requires a callback. Callback state is borrowed for the process
/// lifetime once the returned record reports creation, including an incomplete
/// current-resource switch. A transaction that reaches the registry returns
/// its locked record; identified reuse requires the original callback binding.
/// A failure before registry access leaves the record zeroed (`struct_size=0`).
cudf_status_t cudf_memory_manager_install(
  int32_t device,
  cudf_memory_manager_options_t const* options,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudf_memory_manager_installation_identity_t const* identity,
  cudf_memory_manager_handle* out,
  cudf_memory_manager_installation_record_t* installation);
cudf_status_t cudf_memory_manager_installation_record(
  int32_t device, cudf_memory_manager_installation_record_t* out);
cudf_status_t cudf_memory_manager_failure_callback_state(int32_t device,
                                                         int32_t* exists,
                                                         int32_t* failure_callback_installed);
cudf_status_t cudf_memory_manager_installation_state(int32_t device,
                                                     int32_t* exists,
                                                     int32_t* failure_callback_installed,
                                                     int32_t* current_resource_installed);
cudf_status_t cudf_memory_manager_options_for_device(int32_t device,
                                                     int32_t* exists,
                                                     cudf_memory_manager_options_t* out);
cudf_status_t cudf_memory_manager_resource(cudf_memory_manager_handle manager,
                                           cudf_memory_resource_handle* out);
/// Route global/current-RMM allocations issued on `stream` through `resource`.
///
/// The manager's installed current resource remains unchanged. The returned
/// lease retains the target resource until it is destroyed; allocations made
/// through the lease retain their route until their matching deallocation.
/// When `resource` is already this manager's permanent resource, `*out` is
/// null and no route is needed.
cudf_status_t cudf_memory_manager_route_current_resource(cudf_memory_manager_handle manager,
                                                         cudf_stream_handle stream,
                                                         cudf_memory_resource_handle resource,
                                                         cudf_current_resource_route_handle* out);
cudf_status_t cudf_current_resource_route_destroy(cudf_current_resource_route_handle route);
cudf_status_t cudf_memory_manager_pool_usage(cudf_memory_manager_handle manager,
                                             cudf_memory_pool_usage_t* out);
cudf_status_t cudf_memory_manager_trim_to(cudf_memory_manager_handle manager, uint64_t keep_bytes);
/// Reads the current fixed-ring bounds. This is a consumer operation; allocator
/// callbacks never take its registry lock.
cudf_status_t cudf_memory_manager_allocator_event_ring_bounds(
  cudf_memory_manager_handle manager, cudf_allocator_event_ring_bounds_t* out);
/// Reads one exact sequence. `out` is zeroed when the result is not available.
cudf_status_t cudf_memory_manager_read_allocator_event(cudf_memory_manager_handle manager,
                                                       uint64_t sequence,
                                                       cudf_allocator_event_t* out,
                                                       cudf_allocator_event_read_status_t* status);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_MEMORY_MANAGER_H
