use std::os::raw::{c_char, c_void};

use crate::{
    cudf_ast_expression_handle, cudf_exec_options_t, cudf_status_t, cudf_table_handle, cudf_table_view_handle,
};

extern "C" {
    // -- I/O: Parquet --------------------------------------------------------

    /// Read Parquet with explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned table.
    pub fn cudf_io_read_parquet_ex(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Read Parquet from a host-memory buffer with explicit execution context.
    ///
    /// `host_buffer` is borrowed only for the duration of this call. `options`
    /// must name an explicit valid device and non-null stream/MR.
    pub fn cudf_io_read_parquet_from_host_buffer_ex(
        host_buffer: *const c_void,
        host_buffer_len: usize,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Read Parquet with metadata and explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned table.
    pub fn cudf_io_read_parquet_with_metadata_ex(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Read Parquet with metadata, explicit execution context, and an optional
    /// libcudf AST reader predicate.
    ///
    /// Passing null for `predicate` preserves ordinary unfiltered reader
    /// behavior. `options` must name an explicit valid device and non-null
    /// stream/MR. `options->mr` must outlive the returned table.
    pub fn cudf_io_read_parquet_with_metadata_filter_ex(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        predicate: cudf_ast_expression_handle,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Read Parquet from a host-memory buffer with metadata, explicit
    /// execution context, and an optional libcudf AST reader predicate.
    ///
    /// Passing null for `predicate` preserves ordinary unfiltered reader
    /// behavior. `host_buffer` is borrowed only for the duration of this call.
    pub fn cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex(
        host_buffer: *const c_void,
        host_buffer_len: usize,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        predicate: cudf_ast_expression_handle,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Free a heap-allocated array of host column names returned by this shim.
    ///
    /// When the associated count is zero, the shim returns a null pointer and
    /// this function may be skipped.
    pub fn cudf_io_free_column_names(names: *mut *mut c_char, num_names: i32);

    /// Write Parquet with explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// The shim installs `options->mr` as the current device resource only for
    /// the duration of the libcudf writer call.
    pub fn cudf_io_write_parquet_ex(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        row_group_size_rows: i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
