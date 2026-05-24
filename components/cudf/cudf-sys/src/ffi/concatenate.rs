use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_status_t, cudf_table_handle,
    cudf_table_view_handle,
};

extern "C" {
    // -- Concatenate ---------------------------------------------------------

    /// Concatenate multiple columns into one with explicit execution context.
    ///
    /// `options` must be non-null and carry explicit device, stream, and
    /// memory-resource handles. `options->mr` must outlive the returned column.
    pub fn cudf_concatenate_columns_ex(
        columns: *const cudf_column_view_handle,
        num_columns: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Concatenate multiple tables into one with explicit execution context.
    ///
    /// `options` must be non-null and carry explicit device, stream, and
    /// memory-resource handles. `options->mr` must outlive the returned table.
    pub fn cudf_concatenate_tables_ex(
        tables: *const cudf_table_view_handle,
        num_tables: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;
}
