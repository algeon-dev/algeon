use std::os::raw::{c_char, c_void};

use crate::{cudf_exec_options_t, cudf_scalar_handle, cudf_status_t, cudf_type_id_t};

extern "C" {
    // -- Scalar --------------------------------------------------------------

    /// Create a numeric scalar on cuDF's default stream without an extra
    /// synchronization barrier before returning.
    pub fn cudf_scalar_create_numeric(
        type_id: cudf_type_id_t,
        value: *const c_void,
        is_valid: i32,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a numeric scalar with explicit execution options.
    ///
    /// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
    /// NULL, uses cuDF's default stream and current device memory resource.
    /// This helper does not add an extra stream synchronization barrier before
    /// returning.
    /// When `options->mr` is non-null, it must outlive the returned scalar.
    pub fn cudf_scalar_create_numeric_ex(
        type_id: cudf_type_id_t,
        value: *const c_void,
        is_valid: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a decimal scalar from its raw integer representation.
    ///
    /// `type_id` must be DECIMAL32, DECIMAL64, or DECIMAL128. `value` points
    /// to an i32, i64, or i128 raw representation, respectively. Invalid
    /// scalars may pass a null `value`.
    pub fn cudf_scalar_create_decimal_raw(
        type_id: cudf_type_id_t,
        value: *const c_void,
        scale: i32,
        is_valid: i32,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a decimal scalar from its raw integer representation with
    /// explicit execution options.
    pub fn cudf_scalar_create_decimal_raw_ex(
        type_id: cudf_type_id_t,
        value: *const c_void,
        scale: i32,
        is_valid: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a timestamp scalar from a raw integer count.
    pub fn cudf_scalar_create_timestamp_raw(
        type_id: cudf_type_id_t,
        value: i64,
        is_valid: i32,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a timestamp scalar from a raw integer count with explicit
    /// execution options.
    pub fn cudf_scalar_create_timestamp_raw_ex(
        type_id: cudf_type_id_t,
        value: i64,
        is_valid: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a duration scalar from a raw integer count.
    pub fn cudf_scalar_create_duration_raw(
        type_id: cudf_type_id_t,
        value: i64,
        is_valid: i32,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a duration scalar from a raw integer count with explicit
    /// execution options.
    pub fn cudf_scalar_create_duration_raw_ex(
        type_id: cudf_type_id_t,
        value: i64,
        is_valid: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a string scalar on cuDF's default stream.
    ///
    /// This helper synchronizes the default stream before return on success
    /// so the borrowed host string is no longer in use.
    pub fn cudf_scalar_create_string(
        value: *const c_char,
        is_valid: i32,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a string scalar with explicit execution options.
    ///
    /// The selected stream is synchronized before return on success so the
    /// borrowed host string is no longer in use.
    /// When `options->mr` is non-null, it must outlive the returned scalar.
    pub fn cudf_scalar_create_string_ex(
        value: *const c_char,
        is_valid: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Deep-copy a scalar on cuDF's default stream.
    ///
    /// Supports all scalar subtypes except DICTIONARY32.
    /// This is a device-only operation with no stream synchronization.
    pub fn cudf_scalar_copy(src: cudf_scalar_handle, out: *mut cudf_scalar_handle) -> cudf_status_t;

    /// Deep-copy a scalar with explicit execution options.
    ///
    /// When `options` is null, uses cuDF's default stream and memory resource.
    /// This is a device-only operation with no stream synchronization.
    pub fn cudf_scalar_copy_ex(
        src: cudf_scalar_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    pub fn cudf_scalar_destroy(handle: cudf_scalar_handle);

    pub fn cudf_scalar_type(handle: cudf_scalar_handle, out: *mut cudf_type_id_t) -> cudf_status_t;

    pub fn cudf_scalar_data_type(
        handle: cudf_scalar_handle,
        out_type_id: *mut cudf_type_id_t,
        out_scale: *mut i32,
    ) -> cudf_status_t;

    pub fn cudf_scalar_is_valid(handle: cudf_scalar_handle, out: *mut i32) -> cudf_status_t;

    /// Check scalar validity with explicit execution options.
    ///
    /// On success, `out` is ready to read when this call returns because
    /// libcudf performs synchronous host materialization internally.
    /// `options->mr` is ignored because this helper does not allocate device
    /// outputs, and this shim does not add an extra synchronization barrier
    /// beyond that materialization.
    pub fn cudf_scalar_is_valid_ex(
        handle: cudf_scalar_handle,
        out: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_scalar_value(handle: cudf_scalar_handle, out: *mut c_void) -> cudf_status_t;

    /// Materialize a fixed-width scalar raw value with explicit execution
    /// options.
    ///
    /// On success, `out` is ready to read when this call returns because
    /// libcudf performs synchronous host materialization internally.
    /// `options->mr` is ignored because this helper does not allocate device
    /// outputs, and this shim does not add an extra synchronization barrier
    /// beyond that materialization.
    pub fn cudf_scalar_value_ex(
        handle: cudf_scalar_handle,
        out: *mut c_void,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Materialize a string scalar into an owned host buffer.
    ///
    /// On success, `out` receives a heap-allocated, null-terminated UTF-8
    /// string and `len_out` receives its byte length excluding the terminator.
    /// The caller must release `out` with [`cudf_string_free`].
    pub fn cudf_scalar_string_value(
        handle: cudf_scalar_handle,
        out: *mut *mut c_char,
        len_out: *mut i32,
    ) -> cudf_status_t;

    /// Materialize a string scalar into an owned host buffer with explicit
    /// execution options.
    ///
    /// On success, `out` receives a heap-allocated, null-terminated UTF-8
    /// string and `len_out` receives its byte length excluding the terminator.
    /// The caller must release `out` with [`cudf_string_free`]. The owned host
    /// copy is ready to read when this call returns because libcudf performs
    /// synchronous host materialization internally. `options->mr` is ignored
    /// because this helper does not allocate device outputs, and this shim
    /// does not add an extra synchronization barrier beyond that
    /// materialization.
    pub fn cudf_scalar_string_value_ex(
        handle: cudf_scalar_handle,
        out: *mut *mut c_char,
        len_out: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Free an owned host string returned by this shim.
    pub fn cudf_string_free(ptr: *mut c_char);
}
