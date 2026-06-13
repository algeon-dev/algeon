use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_interop_buffer_binding_t,
    cudf_interop_semantic_column_t, cudf_interop_semantic_table_t, cudf_status_t, cudf_table_handle,
    cudf_table_view_handle,
};

extern "C" {
    // -- Table (owning) ------------------------------------------------------

    /// Create a table from owning column handles.
    ///
    /// On success, all input column handles are consumed and moved into the
    /// returned table.
    ///
    /// On failure, no input columns are consumed and the caller retains
    /// ownership of every input handle.
    pub fn cudf_table_create(
        columns: *mut cudf_column_handle,
        num_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy(src: cudf_table_handle, out: *mut cudf_table_handle) -> cudf_status_t;

    pub fn cudf_table_copy_ex(
        src: cudf_table_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy_from_prepared_device_export(
        table: *const cudf_interop_semantic_table_t,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy_from_prepared_device_export_ex(
        table: *const cudf_interop_semantic_table_t,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy_from_prepared_host_export(
        table: *const cudf_interop_semantic_table_t,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy_from_prepared_host_export_ex(
        table: *const cudf_interop_semantic_table_t,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_create_from_owned_device_transfer(
        table: *const cudf_interop_semantic_table_t,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_payload_size(handle: cudf_table_handle, out: *mut usize) -> cudf_status_t;

    pub fn cudf_table_payload_size_ex(
        handle: cudf_table_handle,
        options: *const cudf_exec_options_t,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_table_device_allocated_size(handle: cudf_table_handle, out: *mut usize) -> cudf_status_t;

    pub fn cudf_table_destroy(handle: cudf_table_handle);

    // -- TableView (non-owning) ----------------------------------------------

    pub fn cudf_table_view(handle: cudf_table_handle, out: *mut cudf_table_view_handle) -> cudf_status_t;

    pub fn cudf_table_view_destroy(handle: cudf_table_view_handle);

    pub fn cudf_table_view_num_columns(handle: cudf_table_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_table_view_num_rows(handle: cudf_table_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_table_view_column(
        handle: cudf_table_view_handle,
        index: i32,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_view_from_column_views(
        column_views: *mut cudf_column_view_handle,
        num_columns: i32,
        out: *mut cudf_table_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_view_select_columns(
        handle: cudf_table_view_handle,
        indices: *const i32,
        num_indices: i32,
        out: *mut cudf_table_view_handle,
    ) -> cudf_status_t;

    /// Release all columns from a table, destroying the table.
    ///
    /// On success, `table` is consumed and must not be used again.
    /// `out_columns` receives a heap-allocated array of `out_count` column
    /// handles. Each must be destroyed with `cudf_column_destroy`, and the
    /// array freed with `cudf_free_column_array`.
    ///
    /// On failure, including output-array allocation failure, `table` is NOT
    /// consumed and remains owned by the caller.
    pub fn cudf_table_release_columns(
        table: cudf_table_handle,
        out_columns: *mut *mut cudf_column_handle,
        out_count: *mut i32,
    ) -> cudf_status_t;
}
