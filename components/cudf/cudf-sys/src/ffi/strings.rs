use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_scalar_handle, cudf_status_t,
    cudf_strip_side_t,
};

extern "C" {
    // -- String operations ---------------------------------------------------

    /// Convert string column to lowercase.
    pub fn cudf_strings_to_lower(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    /// Convert string column to lowercase with explicit execution context.
    pub fn cudf_strings_to_lower_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Convert string column to uppercase.
    pub fn cudf_strings_to_upper(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    /// Convert string column to uppercase with explicit execution context.
    pub fn cudf_strings_to_upper_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Element-wise literal substring containment test. Returns BOOL8 column.
    pub fn cudf_strings_contains(
        input: cudf_column_view_handle,
        target: cudf_scalar_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Element-wise literal substring containment with explicit execution context.
    pub fn cudf_strings_contains_ex(
        input: cudf_column_view_handle,
        target: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Element-wise starts-with test against a scalar string. Returns BOOL8 column.
    pub fn cudf_strings_starts_with(
        input: cudf_column_view_handle,
        target: cudf_scalar_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Element-wise starts-with with explicit execution context.
    pub fn cudf_strings_starts_with_ex(
        input: cudf_column_view_handle,
        target: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Element-wise ends-with test against a scalar string. Returns BOOL8 column.
    pub fn cudf_strings_ends_with(
        input: cudf_column_view_handle,
        target: cudf_scalar_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Element-wise ends-with with explicit execution context.
    pub fn cudf_strings_ends_with_ex(
        input: cudf_column_view_handle,
        target: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Strip characters from one or both sides of each string.
    pub fn cudf_strings_strip(
        input: cudf_column_view_handle,
        side: cudf_strip_side_t,
        to_strip: cudf_scalar_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Strip with explicit execution context.
    pub fn cudf_strings_strip_ex(
        input: cudf_column_view_handle,
        side: cudf_strip_side_t,
        to_strip: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Character count (not byte count) per string. Returns INT32 column.
    pub fn cudf_strings_count_characters(input: cudf_column_view_handle, out: *mut cudf_column_handle)
    -> cudf_status_t;

    /// Character count with explicit execution context.
    pub fn cudf_strings_count_characters_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Byte count per string. Returns INT32 column.
    pub fn cudf_strings_count_bytes(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    /// Byte count with explicit execution context.
    pub fn cudf_strings_count_bytes_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
