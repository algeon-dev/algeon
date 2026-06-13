use std::os::raw::c_void;

use crate::{cudf_memory_resource_handle, cudf_status_t, cudf_stream_handle};

extern "C" {
    // -- Test helpers --------------------------------------------------------

    /// Force the next `cudf_table_release_columns` handle-array allocation to
    /// fail for regression testing.
    pub fn cudf_test_fail_next_table_release_columns_handle_array_alloc();

    /// Return the raw `uintptr_t` value of the last stream passed to an `_ex`
    /// shim.
    pub fn cudf_test_exec_last_stream_value(out: *mut usize) -> cudf_status_t;

    /// Return the raw `uintptr_t` value of the last stream that was
    /// synchronized by an `_ex` shim (host-materializing ops).
    pub fn cudf_test_exec_last_sync_stream_value(out: *mut usize) -> cudf_status_t;

    /// Return the raw pointer of the last memory resource passed to an `_ex`
    /// shim.
    pub fn cudf_test_exec_last_mr_raw(out: *mut *mut c_void) -> cudf_status_t;

    /// Return the CUDA device ordinal that was current when the last `_ex`
    /// shim resolved execution options, or -1 if no observation was recorded.
    pub fn cudf_test_exec_last_device(out: *mut i32) -> cudf_status_t;

    /// Return the raw `uintptr_t` value of the process-global default stream.
    pub fn cudf_test_default_stream_value(out: *mut usize) -> cudf_status_t;

    /// Convert a `cudf_stream_handle` to its raw `uintptr_t` value.
    pub fn cudf_test_stream_handle_value(handle: cudf_stream_handle, out: *mut usize) -> cudf_status_t;

    /// Return the raw pointer of the process-global current device memory
    /// resource.
    pub fn cudf_test_current_device_resource_raw(out: *mut *mut c_void) -> cudf_status_t;

    /// Reset row-predicate/filter observation counters.
    pub fn cudf_test_row_predicate_reset_observation();

    /// Number of fused row-predicate filter invocations.
    pub fn cudf_test_row_predicate_filter_call_count(out: *mut i32) -> cudf_status_t;

    /// Number of predicate-mask computations performed by the fused path.
    pub fn cudf_test_row_predicate_compute_mask_call_count(out: *mut i32) -> cudf_status_t;

    /// Number of final table-filter applications performed by the fused path.
    pub fn cudf_test_row_predicate_final_filter_call_count(out: *mut i32) -> cudf_status_t;

    /// Number of direct `cudf_apply_boolean_mask[_ex]` API invocations.
    pub fn cudf_test_apply_boolean_mask_call_count(out: *mut i32) -> cudf_status_t;

    /// Create a statistics-wrapped memory resource for allocator-sensitive
    /// integration tests.
    pub fn cudf_test_memory_resource_create_stats(out: *mut cudf_memory_resource_handle) -> cudf_status_t;

    /// Read the cumulative allocation count from a statistics memory resource.
    pub fn cudf_test_memory_resource_allocation_count(
        handle: cudf_memory_resource_handle,
        out: *mut i64,
    ) -> cudf_status_t;

    /// Resolve a memory-resource handle to its underlying `device_memory_resource*`.
    pub fn cudf_test_memory_resource_raw(handle: cudf_memory_resource_handle, out: *mut *mut c_void) -> cudf_status_t;

    /// Install the memory resource owned by `handle` as the process-global
    /// current device resource for tests.
    pub fn cudf_test_memory_resource_set_current(handle: cudf_memory_resource_handle) -> cudf_status_t;

    /// Install a previously observed raw device resource pointer as the
    /// process-global current device resource for tests.
    pub fn cudf_test_memory_resource_set_current_raw(raw_mr: *mut c_void) -> cudf_status_t;

    /// Throw a non-I/O `std::runtime_error` from the native shim for error
    /// classification tests.
    pub fn cudf_test_throw_runtime_error() -> cudf_status_t;

    /// Throw `std::bad_alloc` from the native shim for OOM classification
    /// tests.
    pub fn cudf_test_throw_bad_alloc() -> cudf_status_t;

    /// Throw `std::invalid_argument` from the native shim for invalid-argument
    /// classification tests.
    pub fn cudf_test_throw_invalid_argument() -> cudf_status_t;
}
