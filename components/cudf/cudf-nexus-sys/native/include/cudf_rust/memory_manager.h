#ifndef CUDF_RUST_MEMORY_MANAGER_H
#define CUDF_RUST_MEMORY_MANAGER_H

#include "memory_resource.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to the process-local memory manager for one CUDA device.
typedef struct cudf_memory_manager_handle_state* cudf_memory_manager_handle;

enum { CUDF_MEMORY_MANAGER_OPTIONS_RESERVED_SLOTS = 4 };

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
  uint64_t reserved[CUDF_MEMORY_MANAGER_OPTIONS_RESERVED_SLOTS];
} cudf_memory_manager_options_t;

cudf_status_t cudf_memory_manager_for_device(int32_t device, cudf_memory_manager_handle* out);
cudf_status_t cudf_memory_manager_for_device_with_options(
  int32_t device, cudf_memory_manager_options_t const* options, cudf_memory_manager_handle* out);
cudf_status_t cudf_memory_manager_for_device_with_failure_callback(
  int32_t device,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudf_memory_manager_handle* out);
cudf_status_t cudf_memory_manager_for_device_with_failure_callback_v2(
  int32_t device,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
  cudf_memory_manager_handle* out);
cudf_status_t cudf_memory_manager_for_device_with_failure_callback_and_options(
  int32_t device,
  cudf_memory_failure_callback_t callback,
  void* callback_arg,
  cudf_memory_manager_options_t const* options,
  cudf_memory_manager_handle* out);
cudf_status_t cudf_memory_manager_for_device_with_failure_callback_v2_and_options(
  int32_t device,
  cudf_memory_failure_callback_v2_t callback,
  void* callback_arg,
  cudf_memory_manager_options_t const* options,
  cudf_memory_manager_handle* out);
cudf_status_t cudf_memory_manager_failure_callback_state(int32_t device,
                                                         int32_t* exists,
                                                         int32_t* failure_callback_installed);
cudf_status_t cudf_memory_manager_options_for_device(int32_t device,
                                                     int32_t* exists,
                                                     cudf_memory_manager_options_t* out);
cudf_status_t cudf_memory_manager_resource(cudf_memory_manager_handle manager,
                                           cudf_memory_resource_handle* out);
cudf_status_t cudf_memory_manager_pool_usage(cudf_memory_manager_handle manager,
                                             cudf_memory_pool_usage_t* out);
cudf_status_t cudf_memory_manager_trim_to(cudf_memory_manager_handle manager, uint64_t keep_bytes);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_MEMORY_MANAGER_H
