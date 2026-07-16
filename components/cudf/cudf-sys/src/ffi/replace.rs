use crate::{cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_scalar_handle, cudf_status_t};

unsafe extern "C" {
    // -- Replace -------------------------------------------------------------

    /// Replace nulls from a scalar with explicit execution context.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_replace_nulls_scalar_ex(
        input: cudf_column_view_handle,
        replacement: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Replace nulls from another column with explicit execution context.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_replace_nulls_column_ex(
        input: cudf_column_view_handle,
        replacement: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Replace nulls using a fill policy with explicit execution context.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned column.
    pub fn cudf_replace_nulls_policy_ex(
        input: cudf_column_view_handle,
        policy: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
