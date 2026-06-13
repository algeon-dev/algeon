use crate::{cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_status_t};

extern "C" {
    // -- Search --------------------------------------------------------------

    pub fn cudf_contains(
        haystack: cudf_column_view_handle,
        needles: cudf_column_view_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_contains_ex(
        haystack: cudf_column_view_handle,
        needles: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
