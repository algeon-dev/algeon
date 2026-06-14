use std::os::raw::c_void;

use crate::{cudf_column_handle, cudf_device_buffer_handle, cudf_exec_options_t, cudf_status_t, cudf_type_id_t};

extern "C" {
    // -- DeviceBuffer (owned GPU memory) -------------------------------------

    /// Allocate a device buffer of `size` bytes using the current per-device
    /// RMM memory resource. Contents are uninitialized.
    pub fn cudf_device_buffer_create(size: usize, out: *mut cudf_device_buffer_handle) -> cudf_status_t;

    /// Allocate a device buffer of `size` bytes using the explicit stream and
    /// memory resource in `options`. Contents are uninitialized.
    pub fn cudf_device_buffer_create_ex(
        size: usize,
        options: *const cudf_exec_options_t,
        out: *mut cudf_device_buffer_handle,
    ) -> cudf_status_t;

    /// Release a device buffer. Safe to call with null (no-op).
    /// Must NOT be called if the buffer was previously adopted by a column.
    pub fn cudf_device_buffer_destroy(handle: cudf_device_buffer_handle);

    /// Get the device pointer from a buffer handle.
    ///
    /// For zero-size buffers this call still succeeds, but the returned
    /// pointer may be null because there is no addressable device range.
    pub fn cudf_device_buffer_data(handle: cudf_device_buffer_handle, out: *mut *mut c_void) -> cudf_status_t;

    /// Get the size in bytes of a buffer.
    pub fn cudf_device_buffer_size(handle: cudf_device_buffer_handle, out: *mut usize) -> cudf_status_t;

    /// Copy host data into the device buffer. Synchronizes the default stream
    /// before return.
    pub fn cudf_device_buffer_copy_from_host(
        handle: cudf_device_buffer_handle,
        host_data: *const c_void,
        size: usize,
    ) -> cudf_status_t;

    /// Create a column by adopting (moving) an existing device buffer.
    /// Zero-copy: no device-to-device memcpy occurs.
    ///
    /// On success, the buffer handle is consumed and must NOT be destroyed.
    /// On failure, the buffer handle remains valid.
    pub fn cudf_column_adopt_device_buffer(
        type_id: cudf_type_id_t,
        buffer: cudf_device_buffer_handle,
        row_count: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a nullable column by adopting data and null-mask buffers.
    /// Zero-copy: no device-to-device memcpy occurs.
    ///
    /// On success, both buffer handles are consumed and must NOT be destroyed.
    /// On failure, both remain valid.
    pub fn cudf_column_adopt_device_buffer_nullable(
        type_id: cudf_type_id_t,
        data_buffer: cudf_device_buffer_handle,
        null_mask_buffer: cudf_device_buffer_handle,
        null_count: i32,
        row_count: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;
}
