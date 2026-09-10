#ifndef CUDF_RUST_TEST_SUPPORT_H
#define CUDF_RUST_TEST_SUPPORT_H

/// Test-only C ABI for cudf-rust integration and failure-path coverage.
///
/// This header is intentionally excluded from cudf_rust.h. Consumers must opt
/// into test hooks at build time and include this header explicitly.

#include "cudf_rust.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUDF_TEST_IO_METADATA_FAULT_NONE                = 0,
  CUDF_TEST_IO_METADATA_FAULT_NEGATIVE_NAME_COUNT = 1,
  CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ARRAY     = 2,
  CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ENTRY     = 3,
  CUDF_TEST_IO_METADATA_FAULT_INVALID_UTF8_NAME   = 4,
  CUDF_TEST_IO_METADATA_FAULT_ZERO_NAME_COUNT     = 5,
} cudf_test_io_metadata_fault_t;

void cudf_test_fail_next_table_release_columns_handle_array_alloc(void);
void cudf_test_fail_column_handle_release_after(int32_t successful_releases);
cudf_status_t cudf_test_checked_owning_handle_size_add(size_t left, size_t right, size_t* out);
cudf_status_t cudf_test_checked_owning_handle_size_mul(size_t left, size_t right, size_t* out);
void cudf_test_fail_next_provenance_destroy(cudf_status_t status);

void cudf_test_exec_reset_observation(void);
cudf_status_t cudf_test_exec_last_stream_value(uintptr_t* out);
cudf_status_t cudf_test_exec_last_sync_stream_value(uintptr_t* out);
cudf_status_t cudf_test_exec_last_sync_sequence(uint64_t* out);
cudf_status_t cudf_test_exec_last_memory_resource_destroy_raw(void** out);
cudf_status_t cudf_test_exec_last_memory_resource_destroy_sequence(uint64_t* out);
cudf_status_t cudf_test_exec_last_mr_raw(void** out);
cudf_status_t cudf_test_exec_observed_mr_count(size_t* out);
cudf_status_t cudf_test_exec_observed_mr_overflowed(int32_t* out);
cudf_status_t cudf_test_exec_observed_mr_raw_at(size_t index, void** out);
cudf_status_t cudf_test_exec_last_device(int32_t* out);
cudf_status_t cudf_test_default_stream_value(uintptr_t* out);
cudf_status_t cudf_test_stream_handle_value(cudf_stream_handle handle, uintptr_t* out);
cudf_status_t cudf_test_current_device_resource_raw(void** out);
void cudf_test_memory_manager_fail_next_current_resource_install(void);
void cudf_test_memory_manager_invoke_failure_callback_before_install_returns(void);

void cudf_test_row_predicate_reset_observation(void);
cudf_status_t cudf_test_row_predicate_filter_call_count(int32_t* out);
cudf_status_t cudf_test_row_predicate_compute_mask_call_count(int32_t* out);
cudf_status_t cudf_test_row_predicate_final_filter_call_count(int32_t* out);
cudf_status_t cudf_test_apply_boolean_mask_call_count(int32_t* out);

void cudf_test_fail_next_column_view_metadata_read(void);
void cudf_test_fail_next_fixed_width_host_import_after_data_copy(void);
void cudf_test_fail_next_fixed_width_host_import_after_null_mask_copy(void);
void cudf_test_fail_next_strings_host_import_after_offsets_copy(void);
void cudf_test_fail_next_strings_host_import_after_chars_copy(void);
void cudf_test_fail_next_strings_host_import_after_null_mask_copy(void);
void cudf_test_fail_next_strings_host_import_before_make_strings_column(void);
void cudf_test_fail_next_strings_host_import_after_make_strings_column(void);
cudf_status_t cudf_test_column_view_create_synthetic_fixed_width(cudf_type_id_t type_id,
                                                                 int32_t size,
                                                                 int32_t offset,
                                                                 cudf_column_view_handle* out);
cudf_status_t cudf_test_column_view_create_synthetic_string_without_children(
  int32_t size, cudf_column_view_handle* out);

void cudf_test_groupby_fail_next_negative_total_columns(void);
void cudf_test_table_view_fail_next_negative_num_columns(void);
void cudf_test_table_view_fail_next_negative_num_rows(void);

void cudf_test_io_set_next_metadata_fault(cudf_test_io_metadata_fault_t fault);
void cudf_test_io_reset_metadata_fault(void);
cudf_status_t cudf_test_io_kvikio_active_io_begin(void);
cudf_status_t cudf_test_io_kvikio_active_io_end(void);

void cudf_test_arrow_device_array_set_device_type(cudf_arrow_device_array_handle array,
                                                  int32_t device_type);

void cudf_test_event_reset_observation(void);
void cudf_test_event_block_synchronize(void);
void cudf_test_event_unblock_synchronize(void);
void cudf_test_event_fail_synchronize(void);
void cudf_test_event_allow_synchronize(void);
cudf_status_t cudf_test_event_synchronize_call_count(int32_t* out);
cudf_status_t cudf_test_event_synchronize_call_count_for_event_handle_raw(
  uintptr_t event_handle_raw, int32_t* out);
cudf_status_t cudf_test_event_stream_wait_call_count(int32_t* out);
cudf_status_t cudf_test_event_stream_wait_call_count_for_stream_raw(uintptr_t stream_raw,
                                                                    int32_t* out);
cudf_status_t cudf_test_event_stream_wait_call_count_for_event_raw(uintptr_t event_raw,
                                                                   int32_t* out);
cudf_status_t cudf_test_event_last_wait_stream_raw(uintptr_t* out);
cudf_status_t cudf_test_event_last_wait_event_raw(uintptr_t* out);
cudf_status_t cudf_test_event_raw_record_call_count(int32_t* out);
cudf_status_t cudf_test_event_raw_record_call_count_for_stream_raw(uintptr_t stream_raw,
                                                                   int32_t* out);
cudf_status_t cudf_test_event_last_raw_record_stream_raw(uintptr_t* out);
cudf_status_t cudf_test_event_last_raw_record_event_raw(uintptr_t* out);

void cudf_test_stream_synchronize_reset(void);
void cudf_test_stream_block_synchronize(void);
void cudf_test_stream_unblock_synchronize(void);
void cudf_test_stream_fail_synchronize(void);
void cudf_test_stream_allow_synchronize(void);
cudf_status_t cudf_test_stream_synchronize_call_count(int32_t* out);

cudf_status_t cudf_test_memory_resource_create_stats(cudf_memory_resource_handle* out);
cudf_status_t cudf_test_memory_resource_create_fail_first_allocation(
  cudf_memory_resource_handle upstream, cudf_memory_resource_handle* out);
cudf_status_t cudf_test_memory_resource_allocation_count(cudf_memory_resource_handle handle,
                                                         int64_t* out);
cudf_status_t cudf_test_memory_resource_active_allocation_count(cudf_memory_resource_handle handle,
                                                                int64_t* out);
cudf_status_t cudf_test_memory_resource_raw(cudf_memory_resource_handle handle, void** out);
cudf_status_t cudf_test_memory_resource_allocate(cudf_memory_resource_handle handle,
                                                  cudf_stream_handle stream,
                                                  size_t bytes,
                                                  size_t alignment,
                                                  void** out);
cudf_status_t cudf_test_memory_resource_deallocate(cudf_memory_resource_handle handle,
                                                    cudf_stream_handle stream,
                                                    void* ptr,
                                                    size_t bytes,
                                                    size_t alignment);
cudf_status_t cudf_test_memory_resource_allocation_attempt_count(uint64_t* out);
void cudf_test_memory_resource_reset_allocation_attempts(void);
void cudf_test_query_domain_fail_allocation_after(uint64_t domain_id,
                                                  uint64_t successful_allocations);
void cudf_test_query_domain_reset_allocation_failure(void);
cudf_status_t cudf_test_memory_resource_set_current(cudf_memory_resource_handle handle);
cudf_status_t cudf_test_memory_resource_set_current_untracked(cudf_memory_resource_handle handle);
cudf_status_t cudf_test_memory_resource_set_current_raw(void* raw_mr);
cudf_status_t cudf_test_memory_manager_query_domain_count_for_id(
  cudf_memory_manager_handle manager, uint64_t domain_id, size_t* out);
int32_t cudf_test_query_domain_route_pause_arm(uint64_t domain_id);
int32_t cudf_test_query_domain_route_pause_reached(void);
void cudf_test_query_domain_route_pause_release(void);

cudf_status_t cudf_test_throw_runtime_error(void);
cudf_status_t cudf_test_throw_bad_alloc(void);
cudf_status_t cudf_test_throw_invalid_argument(void);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_TEST_SUPPORT_H
