use std::os::raw::c_char;

use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_data_type_handle, cudf_exec_options_t, cudf_scalar_handle,
    cudf_status_t, cudf_strip_side_t,
};

unsafe extern "C" {
    // -- String operations ---------------------------------------------------

    /// Convert string column to lowercase with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_to_lower_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Convert string column to uppercase with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_to_upper_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Element-wise literal substring containment with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_contains_ex(
        input: cudf_column_view_handle,
        target: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Element-wise starts-with with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_starts_with_ex(
        input: cudf_column_view_handle,
        target: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Element-wise ends-with with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_ends_with_ex(
        input: cudf_column_view_handle,
        target: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Element-wise SQL LIKE pattern match with explicit execution context.
    ///
    /// The selected stream is synchronized before return on success so
    /// borrowed `pattern` and `escape` host memory is no longer in use.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_like_ex(
        input: cudf_column_view_handle,
        pattern: *const c_char,
        pattern_len: i32,
        escape: *const c_char,
        escape_len: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Element-wise regex replacement with back-reference templates and explicit execution context.
    ///
    /// The selected stream is synchronized before return on success so
    /// borrowed `pattern` and `replacement` host memory is no longer in use.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_replace_with_backrefs_ex(
        input: cudf_column_view_handle,
        pattern: *const c_char,
        pattern_len: i32,
        replacement: *const c_char,
        replacement_len: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Slice each string with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_slice_ex(
        input: cudf_column_view_handle,
        start: i32,
        start_valid: i32,
        stop: i32,
        stop_valid: i32,
        step: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Row-wise concatenate string columns with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_concatenate_ex(
        columns: *const cudf_column_view_handle,
        num_columns: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Strip with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_strip_ex(
        input: cudf_column_view_handle,
        side: cudf_strip_side_t,
        to_strip: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Character count with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_count_characters_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Byte count with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_count_bytes_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Try-cast strings to integers with explicit execution context.
    /// `options` must name an explicit valid device and non-NULL stream/MR.
    /// `options->mr` must outlive the returned column.
    pub fn cudf_strings_try_to_integers_ex(
        input: cudf_column_view_handle,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
