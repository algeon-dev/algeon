use crate::{cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_status_t};

extern "C" {
    // -- Search --------------------------------------------------------------

    /// Check if elements of `needles` exist in `haystack` using explicit
    /// execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_contains_ex(
        haystack: cudf_column_view_handle,
        needles: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
