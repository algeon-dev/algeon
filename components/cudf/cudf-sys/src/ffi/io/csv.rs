use std::os::raw::c_char;

use crate::{cudf_exec_options_t, cudf_status_t, cudf_table_handle, cudf_table_view_handle};

extern "C" {
    // -- I/O: CSV ------------------------------------------------------------

    /// Read a CSV file into a table with explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned table.
    pub fn cudf_io_read_csv_ex(
        filepath: *const c_char,
        delimiter: c_char,
        has_header: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Write a CSV file with explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// The shim installs `options->mr` as the current device resource only for
    /// the duration of the libcudf writer call.
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
