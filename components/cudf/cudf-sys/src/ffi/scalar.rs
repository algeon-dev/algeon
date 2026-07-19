use std::os::raw::{c_char, c_void};

use crate::{cudf_exec_options_t, cudf_scalar_handle, cudf_status_t, cudf_type_id_t};

unsafe extern "C" {
    // -- Scalar --------------------------------------------------------------

    /// Create a numeric scalar with explicit execution options.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
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

    /// Create a timestamp scalar from a raw integer count with explicit
    /// execution options.
    pub fn cudf_scalar_create_timestamp_raw_ex(
        type_id: cudf_type_id_t,
        value: i64,
        is_valid: i32,
        options: *const cudf_exec_options_t,
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

    /// Deep-copy a scalar with explicit execution options.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// This is a device-only operation with no stream synchronization.
    pub fn cudf_scalar_copy_ex(
        src: cudf_scalar_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    pub fn cudf_scalar_destroy(handle: cudf_scalar_handle) -> cudf_status_t;

    /// `out` receives a raw `cudf_type_id_t` discriminant. Declared as `i32`
    /// so an unmirrored libcudf type id never materializes an invalid Rust
    /// enum; callers must validate before converting.
    pub fn cudf_scalar_type(handle: cudf_scalar_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_scalar_data_type(
        handle: cudf_scalar_handle,
        out_type_id: *mut i32,
        out_scale: *mut i32,
    ) -> cudf_status_t;

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
