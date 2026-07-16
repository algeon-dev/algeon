use std::os::raw::c_char;

use crate::{cudf_exec_options_t, cudf_status_t, cudf_table_handle, cudf_table_view_handle};

unsafe extern "C" {
    // -- JSON I/O ------------------------------------------------------------

    /// Read JSON with explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned table.
    pub fn cudf_io_read_json_ex(
        filepath: *const c_char,
        lines: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
    /// Read JSON with metadata and explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned table.
    pub fn cudf_io_read_json_with_metadata_ex(
        filepath: *const c_char,
        lines: i32,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
    /// Write JSON with explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// The shim installs `options->mr` as the current device resource only for
    /// the duration of the libcudf writer call.
    pub fn cudf_io_write_json_ex(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        lines: i32,
        include_nulls: i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
