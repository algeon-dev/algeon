use std::os::raw::c_char;

use crate::{cudf_exec_options_t, cudf_status_t, cudf_table_handle, cudf_table_view_handle};

extern "C" {
    // -- I/O: CSV ------------------------------------------------------------

    pub fn cudf_io_read_csv(
        filepath: *const c_char,
        delimiter: c_char,
        has_header: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Read a CSV file into a table with explicit execution context.
    ///
    /// Passing null for `options`, or leaving `options->stream` /
    /// `options->mr` null, uses cuDF's default stream and current device
    /// memory resource. When `options->mr` is non-null, it must outlive the
    /// returned table.
    pub fn cudf_io_read_csv_ex(
        filepath: *const c_char,
        delimiter: c_char,
        has_header: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_io_write_csv(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        delimiter: c_char,
        include_header: i32,
    ) -> cudf_status_t;

    pub fn cudf_io_write_csv_ex(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        delimiter: c_char,
        include_header: i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
