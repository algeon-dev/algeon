use crate::{cudf_column_view_handle, cudf_status_t, cudf_type_id_t};

extern "C" {
    // -- Test observation hooks -----------------------------------------------
    //
    // These functions are defined in `native/src/execution_context.cpp` and
    // allow integration tests to verify that explicit CUDA streams and memory
    // resources are correctly propagated through `_ex` FFI entry points.
    //
    // They are always compiled into `libcudf_rust` (no `#ifdef` guard) but are
    // intended only for test use.  They are unconditionally declared here so
    // that downstream crates (e.g. `cudf-rs` integration tests) can import
    // them from `cudf_sys` instead of re-declaring them locally.

    /// Reset the observation state so the next `_ex` call's context is visible.
    pub fn cudf_test_exec_reset_observation();

    /// Force the next column-view metadata read (`size` or `null_count`) to
    /// fail for panic-elimination regression testing.
    pub fn cudf_test_fail_next_column_view_metadata_read();

    /// Force the next fixed-width host import to fail after the data copy is
    /// queued for synchronization regression testing.
    pub fn cudf_test_fail_next_fixed_width_host_import_after_data_copy();

    /// Force the next fixed-width host import to fail after the null-mask copy
    /// is queued for synchronization regression testing.
    pub fn cudf_test_fail_next_fixed_width_host_import_after_null_mask_copy();

    /// Force the next groupby aggregate call to report a negative result
    /// column count on success for foreign-contract regression testing.
    pub fn cudf_test_groupby_fail_next_negative_total_columns();

    /// Force the next table-view metadata read for column count to report a
    /// negative value on success for foreign-contract regression testing.
    pub fn cudf_test_table_view_fail_next_negative_num_columns();

    /// Force the next table-view metadata read for row count to report a
    /// negative value on success for foreign-contract regression testing.
    pub fn cudf_test_table_view_fail_next_negative_num_rows();

    /// Arm a one-shot malformed metadata response for the next JSON/Parquet
    /// metadata read on the current thread.
    pub fn cudf_test_io_set_next_metadata_fault(fault: i32);

    /// Clear any armed malformed metadata response on the current thread.
    pub fn cudf_test_io_reset_metadata_fault();

    /// Construct a synthetic fixed-width column_view with a caller-controlled
    /// logical offset. Intended only for regression tests that need edge-case
    /// offset values without materializing massive backing buffers.
    pub fn cudf_test_column_view_create_synthetic_fixed_width(
        type_id: cudf_type_id_t,
        size: i32,
        offset: i32,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    /// Construct a synthetic non-empty STRING column_view without an offsets
    /// child. Intended only for regression tests that validate shim-side
    /// layout checks before dereferencing string children.
    pub fn cudf_test_column_view_create_synthetic_string_without_children(
        size: i32,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;
}
