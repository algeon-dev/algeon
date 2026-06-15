use std::os::raw::c_void;

use crate::{cudf_column_handle, cudf_exec_options_t, cudf_status_t};

extern "C" {
    /// Import one host Arrow C Data Interface column through libcudf's host
    /// Arrow column importer.
    pub fn cudf_arrow_host_column_import_ex(
        schema: *const c_void,
        array: *const c_void,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;
}
