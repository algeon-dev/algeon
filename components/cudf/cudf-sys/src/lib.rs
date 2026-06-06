//! Low-level FFI bindings to the cuDF C shim library (`libcudf_rust`).
//!
//! This crate provides raw `extern "C"` function declarations matching the
//! C shim layer that wraps cuDF's C++ API. It is not intended to be used
//! directly — use the `cudf` crate for a safe, idiomatic Rust API.
//!
//! Stream and memory-resource handles can be attached to a size-tagged
//! execution contract (`cudf_exec_options_t`) for `_ex` entry points.
//! Passing a null options pointer, or null handles inside the struct, keeps the
//! legacy behavior of using cuDF's default stream and current device memory
//! resource.

#![allow(non_camel_case_types)]

use std::{
    marker::PhantomData,
    os::raw::{c_char, c_void},
};

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

pub const CUDF_SUCCESS: i32 = 0;
pub const CUDF_ERROR_INVALID_OPERATION: i32 = 1;
pub const CUDF_ERROR_CUDA: i32 = 2;
pub const CUDF_ERROR_OUT_OF_MEMORY: i32 = 3;
pub const CUDF_ERROR_IO: i32 = 4;
pub const CUDF_ERROR_TYPE: i32 = 5;
pub const CUDF_ERROR_NULL: i32 = 6;
pub const CUDF_ERROR_INDEX: i32 = 7;
pub const CUDF_ERROR_UNKNOWN: i32 = 8;
pub const CUDF_ERROR_FATAL_CUDA: i32 = 9;

pub type cudf_status_t = i32;

// ---------------------------------------------------------------------------
// Type IDs (mirrors cudf::type_id)
//
// FFI enums use `#[repr(i32)]` to make the Rust-side ABI explicitly 32-bit.
// The C/C++ shim must preserve that contract and validates enum sizes and
// discriminants in `native/src/abi_checks.cpp`. Update those checks whenever
// changing an FFI enum.
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_type_id_t {
    CUDF_TYPE_EMPTY = 0,
    CUDF_TYPE_INT8,
    CUDF_TYPE_INT16,
    CUDF_TYPE_INT32,
    CUDF_TYPE_INT64,
    CUDF_TYPE_UINT8,
    CUDF_TYPE_UINT16,
    CUDF_TYPE_UINT32,
    CUDF_TYPE_UINT64,
    CUDF_TYPE_FLOAT32,
    CUDF_TYPE_FLOAT64,
    CUDF_TYPE_BOOL8,
    CUDF_TYPE_TIMESTAMP_DAYS,
    CUDF_TYPE_TIMESTAMP_SECONDS,
    CUDF_TYPE_TIMESTAMP_MILLISECONDS,
    CUDF_TYPE_TIMESTAMP_MICROSECONDS,
    CUDF_TYPE_TIMESTAMP_NANOSECONDS,
    CUDF_TYPE_DURATION_DAYS,
    CUDF_TYPE_DURATION_SECONDS,
    CUDF_TYPE_DURATION_MILLISECONDS,
    CUDF_TYPE_DURATION_MICROSECONDS,
    CUDF_TYPE_DURATION_NANOSECONDS,
    CUDF_TYPE_DICTIONARY32,
    CUDF_TYPE_STRING,
    CUDF_TYPE_LIST,
    CUDF_TYPE_DECIMAL32,
    CUDF_TYPE_DECIMAL64,
    CUDF_TYPE_DECIMAL128,
    CUDF_TYPE_STRUCT,
}

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

pub type cudf_data_type_handle = *mut c_void;
pub type cudf_column_handle = *mut c_void;
pub type cudf_column_view_handle = *mut c_void;
pub type cudf_table_handle = *mut c_void;
pub type cudf_table_view_handle = *mut c_void;
pub type cudf_scalar_handle = *mut c_void;
pub type cudf_device_buffer_handle = *mut c_void;
pub type cudf_stream_handle = *mut c_void;
pub type cudf_memory_resource_handle = *mut c_void;

// ---------------------------------------------------------------------------
// Execution options and typed handle wrappers
// ---------------------------------------------------------------------------

pub const CUDF_EXEC_OPTIONS_RESERVED_SLOTS: usize = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_exec_options_t {
    pub struct_size: usize,
    pub stream: cudf_stream_handle,
    pub mr: cudf_memory_resource_handle,
    pub flags: u64,
    pub reserved: [u64; CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
}

impl cudf_exec_options_t {
    pub const fn defaults() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            stream: std::ptr::null_mut(),
            mr: std::ptr::null_mut(),
            flags: 0,
            reserved: [0; CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
        }
    }
}

impl Default for cudf_exec_options_t {
    fn default() -> Self {
        Self::defaults()
    }
}

fn status_to_result(status: cudf_status_t) -> Result<(), cudf_status_t> {
    if status == CUDF_SUCCESS { Ok(()) } else { Err(status) }
}

#[derive(Debug)]
pub struct Stream {
    handle: cudf_stream_handle,
}

impl Stream {
    pub fn new() -> Result<Self, cudf_status_t> {
        let mut handle = std::ptr::null_mut();
        status_to_result(unsafe { cudf_stream_create(&mut handle) })?;
        Ok(Self { handle })
    }

    pub fn as_raw(&self) -> cudf_stream_handle {
        self.handle
    }

    pub fn synchronize(&self) -> Result<(), cudf_status_t> {
        status_to_result(unsafe { cudf_stream_synchronize(self.handle) })
    }

    pub fn is_ready(&self) -> Result<bool, cudf_status_t> {
        let mut out = 0;
        status_to_result(unsafe { cudf_stream_is_ready(self.handle, &mut out) })?;
        Ok(out != 0)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cudf_stream_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

#[derive(Debug)]
pub struct MemoryResource {
    handle: cudf_memory_resource_handle,
}

impl MemoryResource {
    pub fn new_cuda() -> Result<Self, cudf_status_t> {
        let mut handle = std::ptr::null_mut();
        status_to_result(unsafe { cudf_memory_resource_create_cuda(&mut handle) })?;
        Ok(Self { handle })
    }

    pub fn new_pool(initial_pool_size: usize, maximum_pool_size: usize) -> Result<Self, cudf_status_t> {
        let mut handle = std::ptr::null_mut();
        status_to_result(unsafe {
            cudf_memory_resource_create_pool(initial_pool_size, maximum_pool_size, &mut handle)
        })?;
        Ok(Self { handle })
    }

    /// Create a statistics-enabled memory resource for allocator-sensitive tests.
    pub fn new_test_stats() -> Result<Self, cudf_status_t> {
        let mut handle = std::ptr::null_mut();
        status_to_result(unsafe { cudf_test_memory_resource_create_stats(&mut handle) })?;
        Ok(Self { handle })
    }

    pub fn as_raw(&self) -> cudf_memory_resource_handle {
        self.handle
    }

    /// Read the cumulative allocation count from a statistics test resource.
    pub fn test_allocation_count(&self) -> Result<i64, cudf_status_t> {
        let mut out = 0_i64;
        status_to_result(unsafe { cudf_test_memory_resource_allocation_count(self.handle, &mut out) })?;
        Ok(out)
    }
}

impl Drop for MemoryResource {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cudf_memory_resource_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Typed builder for `cudf_exec_options_t`.
///
/// Null stream / memory-resource fields preserve cuDF defaults.
/// If an explicit memory resource is attached, it must outlive any outputs
/// allocated through the `_ex` call that consumes this context.
#[derive(Debug, Clone, Copy)]
pub struct ExecutionContext<'a> {
    raw: cudf_exec_options_t,
    _stream: PhantomData<&'a Stream>,
    _mr: PhantomData<&'a MemoryResource>,
}

impl<'a> ExecutionContext<'a> {
    pub fn new() -> Self {
        Self { raw: cudf_exec_options_t::defaults(), _stream: PhantomData, _mr: PhantomData }
    }

    pub fn with_stream(mut self, stream: &'a Stream) -> Self {
        self.raw.stream = stream.as_raw();
        self
    }

    pub fn with_memory_resource(mut self, mr: &'a MemoryResource) -> Self {
        self.raw.mr = mr.as_raw();
        self
    }

    pub fn raw(&self) -> &cudf_exec_options_t {
        &self.raw
    }

    pub fn as_raw(&self) -> *const cudf_exec_options_t {
        self.raw()
    }
}

impl Default for ExecutionContext<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Install `mr` as the process-global current device resource for tests.
pub fn test_set_current_device_resource(mr: &MemoryResource) -> Result<(), cudf_status_t> {
    status_to_result(unsafe { cudf_test_memory_resource_set_current(mr.as_raw()) })
}

/// Restore a previously observed raw device resource pointer for tests.
pub fn test_set_current_device_resource_raw(raw_mr: *mut c_void) -> Result<(), cudf_status_t> {
    status_to_result(unsafe { cudf_test_memory_resource_set_current_raw(raw_mr) })
}

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

extern "C" {
    // -- Error ---------------------------------------------------------------

    pub fn cudf_last_error_message() -> *const c_char;
    pub fn cudf_clear_error();

    // -- DataType ------------------------------------------------------------

    pub fn cudf_data_type_create(type_id: cudf_type_id_t, out: *mut cudf_data_type_handle) -> cudf_status_t;

    pub fn cudf_data_type_create_fixed_point(
        type_id: cudf_type_id_t,
        scale: i32,
        out: *mut cudf_data_type_handle,
    ) -> cudf_status_t;

    pub fn cudf_data_type_destroy(handle: cudf_data_type_handle);

    pub fn cudf_data_type_id(handle: cudf_data_type_handle, out: *mut cudf_type_id_t) -> cudf_status_t;

    pub fn cudf_data_type_scale(handle: cudf_data_type_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_type_size(type_id: cudf_type_id_t, out: *mut i32) -> cudf_status_t;

    // -- Column (owning) -----------------------------------------------------

    pub fn cudf_column_create_empty(type_id: cudf_type_id_t, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_column_create_numeric(
        type_id: cudf_type_id_t,
        data: *const c_void,
        size: i32,
        null_mask: *const u8,
        null_count: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_create_numeric_ex(
        type_id: cudf_type_id_t,
        data: *const c_void,
        size: i32,
        null_mask: *const u8,
        null_count: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_copy(src: cudf_column_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_column_copy_ex(
        src: cudf_column_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_copy_from_view(src: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_column_copy_from_view_ex(
        src: cudf_column_view_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_destroy(handle: cudf_column_handle);

    // -- ColumnView (non-owning) --------------------------------------------

    pub fn cudf_column_view(handle: cudf_column_handle, out: *mut cudf_column_view_handle) -> cudf_status_t;

    pub fn cudf_column_view_destroy(handle: cudf_column_view_handle);

    pub fn cudf_column_view_type(handle: cudf_column_view_handle, out: *mut cudf_type_id_t) -> cudf_status_t;

    pub fn cudf_column_view_data_type(
        handle: cudf_column_view_handle,
        out_type_id: *mut cudf_type_id_t,
        out_scale: *mut i32,
    ) -> cudf_status_t;

    pub fn cudf_column_view_size(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_column_view_null_count(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_column_view_nullable(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_column_view_num_children(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_column_view_child(
        handle: cudf_column_view_handle,
        index: i32,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_view_data_ptr(handle: cudf_column_view_handle, out: *mut *const c_void) -> cudf_status_t;

    pub fn cudf_column_view_offset(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_column_view_to_host(
        handle: cudf_column_view_handle,
        host_buf: *mut c_void,
        buf_size: usize,
    ) -> cudf_status_t;

    pub fn cudf_column_view_to_host_ex(
        handle: cudf_column_view_handle,
        host_buf: *mut c_void,
        buf_size: usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Materialize a column null mask into a host buffer.
    ///
    /// This helper synchronizes cuDF's default stream before return so
    /// `host_buf` is fully populated on success.
    pub fn cudf_column_view_null_mask_to_host(
        handle: cudf_column_view_handle,
        host_buf: *mut u8,
        buf_size: usize,
    ) -> cudf_status_t;

    /// Materialize a column null mask into a host buffer with explicit
    /// execution options.
    pub fn cudf_column_view_null_mask_to_host_ex(
        handle: cudf_column_view_handle,
        host_buf: *mut u8,
        buf_size: usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Create a column by copying from a borrowed device pointer.
    ///
    /// This legacy helper synchronizes cuDF's default stream before return so
    /// the borrowed `device_data` pointer is no longer in use on success.
    pub fn cudf_column_create_from_device_data(
        type_id: cudf_type_id_t,
        device_data: *const c_void,
        size: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a column by copying from a borrowed device pointer with
    /// explicit execution options.
    ///
    /// Unlike [`cudf_column_create_from_device_data`], this `_ex` variant does
    /// not synchronize before returning. The caller must keep `device_data`
    /// valid until the selected stream has completed the copy and must
    /// establish any required cross-stream ordering for downstream consumers.
    pub fn cudf_column_create_from_device_data_ex(
        type_id: cudf_type_id_t,
        device_data: *const c_void,
        size: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_from_scalar(
        scalar: cudf_scalar_handle,
        size: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a column by repeating a scalar value with explicit execution
    /// options.
    pub fn cudf_column_from_scalar_ex(
        scalar: cudf_scalar_handle,
        size: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_column_create_strings(
        strings: *const *const c_char,
        lengths: *const i32,
        num_strings: i32,
        null_mask: *const u8,
        null_count: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_create_strings_ex(
        strings: *const *const c_char,
        lengths: *const i32,
        num_strings: i32,
        null_mask: *const u8,
        null_count: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Materialize string offsets, and optionally chars, into host buffers.
    ///
    /// This helper synchronizes cuDF's default stream before return whenever
    /// it must populate one of the provided host buffers.
    pub fn cudf_column_view_strings_to_host(
        handle: cudf_column_view_handle,
        offsets_buf: *mut i32,
        offsets_buf_size: usize,
        chars_buf: *mut c_char,
        chars_buf_size: usize,
        total_chars_bytes: *mut i32,
    ) -> cudf_status_t;

    // -- DeviceBuffer (owned GPU memory) -------------------------------------

    /// Allocate a device buffer of `size` bytes using the current per-device
    /// RMM memory resource. Contents are uninitialized.
    pub fn cudf_device_buffer_create(size: usize, out: *mut cudf_device_buffer_handle) -> cudf_status_t;

    /// Release a device buffer. Safe to call with null (no-op).
    /// Must NOT be called if the buffer was previously adopted by a column.
    pub fn cudf_device_buffer_destroy(handle: cudf_device_buffer_handle);

    /// Get the device pointer from a buffer handle.
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

    // -- Table (owning) ------------------------------------------------------

    /// Create a table from owning column handles.
    ///
    /// On success, all input column handles are consumed and moved into the
    /// returned table.
    ///
    /// On failure, no input columns are consumed and the caller retains
    /// ownership of every input handle.
    pub fn cudf_table_create(
        columns: *mut cudf_column_handle,
        num_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy(src: cudf_table_handle, out: *mut cudf_table_handle) -> cudf_status_t;

    pub fn cudf_table_copy_ex(
        src: cudf_table_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_destroy(handle: cudf_table_handle);

    // -- TableView (non-owning) ----------------------------------------------

    pub fn cudf_table_view(handle: cudf_table_handle, out: *mut cudf_table_view_handle) -> cudf_status_t;

    pub fn cudf_table_view_destroy(handle: cudf_table_view_handle);

    pub fn cudf_table_view_num_columns(handle: cudf_table_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_table_view_num_rows(handle: cudf_table_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_table_view_column(
        handle: cudf_table_view_handle,
        index: i32,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_view_from_column_views(
        column_views: *mut cudf_column_view_handle,
        num_columns: i32,
        out: *mut cudf_table_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_view_select_columns(
        handle: cudf_table_view_handle,
        indices: *const i32,
        num_indices: i32,
        out: *mut cudf_table_view_handle,
    ) -> cudf_status_t;

    /// Release all columns from a table, destroying the table.
    ///
    /// On success, `table` is consumed and must not be used again.
    /// `out_columns` receives a heap-allocated array of `out_count` column
    /// handles. Each must be destroyed with `cudf_column_destroy`, and the
    /// array freed with `cudf_free_column_array`.
    ///
    /// On failure, including output-array allocation failure, `table` is NOT
    /// consumed and remains owned by the caller.
    pub fn cudf_table_release_columns(
        table: cudf_table_handle,
        out_columns: *mut *mut cudf_column_handle,
        out_count: *mut i32,
    ) -> cudf_status_t;

    // -- Scalar --------------------------------------------------------------

    /// Create a numeric scalar on cuDF's default stream without an extra
    /// synchronization barrier before returning.
    pub fn cudf_scalar_create_numeric(
        type_id: cudf_type_id_t,
        value: *const c_void,
        is_valid: i32,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// Create a string scalar on cuDF's default stream without an extra
    /// synchronization barrier before returning.
    pub fn cudf_scalar_create_string(
        value: *const c_char,
        is_valid: i32,
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

    pub fn cudf_scalar_value(handle: cudf_scalar_handle, out: *mut c_void) -> cudf_status_t;

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

    /// Free an owned host string returned by this shim.
    pub fn cudf_string_free(ptr: *mut c_char);

    // -- Stream --------------------------------------------------------------

    /// Create an owned CUDA stream handle.
    ///
    /// Attach this handle to [`ExecutionContext`] or [`cudf_exec_options_t`]
    /// when calling `_ex` APIs to direct work onto an explicit stream.
    pub fn cudf_stream_create(out: *mut cudf_stream_handle) -> cudf_status_t;

    /// Destroy a CUDA stream created by [`cudf_stream_create`].
    pub fn cudf_stream_destroy(handle: cudf_stream_handle);

    /// Block until work already enqueued on `handle` completes.
    pub fn cudf_stream_synchronize(handle: cudf_stream_handle) -> cudf_status_t;

    /// Report whether work already enqueued on `handle` has completed.
    pub fn cudf_stream_is_ready(handle: cudf_stream_handle, out: *mut i32) -> cudf_status_t;

    // -- Memory Resource -----------------------------------------------------

    /// Create an owned CUDA-backed RMM memory resource handle.
    ///
    /// Attach this handle to [`ExecutionContext`] or [`cudf_exec_options_t`]
    /// when calling `_ex` APIs to direct allocations through it.
    pub fn cudf_memory_resource_create_cuda(out: *mut cudf_memory_resource_handle) -> cudf_status_t;

    /// Create an owned pool memory resource handle.
    ///
    /// Like [`cudf_memory_resource_create_cuda`], this creates an owned
    /// resource object without changing the process-global allocator.
    pub fn cudf_memory_resource_create_pool(
        initial_pool_size: usize,
        maximum_pool_size: usize,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

    /// Destroy a memory resource created by this shim.
    pub fn cudf_memory_resource_destroy(handle: cudf_memory_resource_handle);

    // -- I/O: Parquet --------------------------------------------------------

    pub fn cudf_io_read_parquet(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_io_read_parquet_ex(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_io_read_parquet_with_metadata(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
    ) -> cudf_status_t;

    pub fn cudf_io_read_parquet_with_metadata_ex(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Free a heap-allocated array of host column names returned by this shim.
    ///
    /// When the associated count is zero, the shim returns a null pointer and
    /// this function may be skipped.
    pub fn cudf_io_free_column_names(names: *mut *mut c_char, num_names: i32);

    pub fn cudf_io_write_parquet(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
    ) -> cudf_status_t;

    /// Write Parquet with explicit execution context.
    /// NOTE: libcudf's write_parquet only accepts stream, not mr.
    /// The mr field in options is ignored for writes.
    pub fn cudf_io_write_parquet_ex(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- I/O: CSV ------------------------------------------------------------

    pub fn cudf_io_read_csv(
        filepath: *const c_char,
        delimiter: c_char,
        has_header: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_io_read_csv_ex(
        filepath: *const c_char,
        delimiter: c_char,
        has_header: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_io_write_csv(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        delimiter: c_char,
        include_header: i32,
    ) -> cudf_status_t;

    pub fn cudf_io_write_csv_ex(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        delimiter: c_char,
        include_header: i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Sorting -------------------------------------------------------------

    pub fn cudf_sort(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_sort_ex(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_stable_sort(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_stable_sort_ex(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_sort_by_key(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_sort_by_key_ex(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_stable_sort_by_key(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_stable_sort_by_key_ex(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_sorted_order(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_sorted_order_ex(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_is_sorted(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut i32,
    ) -> cudf_status_t;

    // -- Filter --------------------------------------------------------------

    pub fn cudf_apply_boolean_mask(
        input: cudf_table_view_handle,
        mask: cudf_column_view_handle,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_apply_boolean_mask_ex(
        input: cudf_table_view_handle,
        mask: cudf_column_view_handle,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- GroupBy --------------------------------------------------------------

    pub fn cudf_aggregation_create(kind: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_count(null_policy: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_std(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_variance(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_destroy(handle: *mut c_void);

    /// Perform a groupby aggregation and return the keys table plus a
    /// heap-allocated flat array of result columns.
    ///
    /// On failure, all outputs are initialized to null / zero. When the total
    /// number of result columns is zero, `out_values` is returned as null.
    pub fn cudf_groupby_aggregate(
        keys: cudf_table_view_handle,
        values: *const cudf_column_view_handle,
        aggregations: *const *mut c_void,
        agg_counts: *const i32,
        num_requests: i32,
        num_aggregations: i32,
        keys_null_policy: i32,
        out_keys: *mut cudf_table_handle,
        out_values: *mut *mut cudf_column_handle,
        out_total_columns: *mut i32,
    ) -> cudf_status_t;

    /// Groupby aggregate with explicit execution context.
    pub fn cudf_groupby_aggregate_ex(
        keys: cudf_table_view_handle,
        values: *const cudf_column_view_handle,
        aggregations: *const *mut c_void,
        agg_counts: *const i32,
        num_requests: i32,
        num_aggregations: i32,
        keys_null_policy: i32,
        out_keys: *mut cudf_table_handle,
        out_values: *mut *mut cudf_column_handle,
        out_total_columns: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Free a heap-allocated array of owning column handles returned by this
    /// shim. Does not destroy the individual columns.
    pub fn cudf_free_column_array(array: *mut cudf_column_handle);

    // -- Unary ---------------------------------------------------------------

    pub fn cudf_unary_operation(input: cudf_column_view_handle, op: i32, out: *mut cudf_column_handle)
    -> cudf_status_t;

    pub fn cudf_is_null(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_is_valid(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_cast(
        input: cudf_column_view_handle,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_is_supported_cast(
        from_type: cudf_data_type_handle,
        to_type: cudf_data_type_handle,
        out: *mut i32,
    ) -> cudf_status_t;

    pub fn cudf_is_nan(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_is_not_nan(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    // -- Binary --------------------------------------------------------------

    pub fn cudf_binary_operation_vv(
        lhs: cudf_column_view_handle,
        rhs: cudf_column_view_handle,
        op: i32,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_binary_operation_vs(
        lhs: cudf_column_view_handle,
        rhs: cudf_scalar_handle,
        op: i32,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_binary_operation_sv(
        lhs: cudf_scalar_handle,
        rhs: cudf_column_view_handle,
        op: i32,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    // -- Reduction -----------------------------------------------------------

    pub fn cudf_reduce_aggregation_create(kind: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_reduce_aggregation_create_variance(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_reduce_aggregation_create_std(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_reduce_aggregation_destroy(handle: *mut c_void);

    pub fn cudf_scan_aggregation_create(kind: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_scan_aggregation_destroy(handle: *mut c_void);

    pub fn cudf_reduce(
        input: cudf_column_view_handle,
        agg: *mut c_void,
        output_type: cudf_data_type_handle,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    pub fn cudf_scan(
        input: cudf_column_view_handle,
        agg: *mut c_void,
        scan_type: i32,
        null_policy: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_minmax(
        input: cudf_column_view_handle,
        out_min: *mut cudf_scalar_handle,
        out_max: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    // -- Concatenate ---------------------------------------------------------

    pub fn cudf_concatenate_columns(
        columns: *const cudf_column_view_handle,
        num_columns: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_concatenate_columns_ex(
        columns: *const cudf_column_view_handle,
        num_columns: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_concatenate_tables(
        tables: *const cudf_table_view_handle,
        num_tables: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_concatenate_tables_ex(
        tables: *const cudf_table_view_handle,
        num_tables: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    // -- Dictionary ----------------------------------------------------------

    pub fn cudf_dictionary_encode(
        input: cudf_column_view_handle,
        indices_type_id: cudf_type_id_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_dictionary_decode(dict_column: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_dictionary_keys(
        dict_column: cudf_column_view_handle,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_dictionary_indices(
        dict_column: cudf_column_view_handle,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_dictionary_set_keys(
        dict_column: cudf_column_view_handle,
        keys: cudf_column_view_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Align multiple dictionary columns to a shared key set.
    ///
    /// On success, `out_columns` receives a heap-allocated array of
    /// `out_count` owning column handles. Destroy each column individually,
    /// then free the array with [`cudf_free_column_array`]. On failure, the
    /// shim initializes `out_columns` to null and `out_count` to zero.
    pub fn cudf_dictionary_match_dictionaries(
        dict_columns: *const cudf_column_view_handle,
        num_columns: i32,
        out_columns: *mut *mut cudf_column_handle,
        out_count: *mut i32,
    ) -> cudf_status_t;

    // -- Hashing -------------------------------------------------------------

    pub fn cudf_hashing_murmurhash3_x86_32(
        input: cudf_table_view_handle,
        seed: u32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hashing_xxhash_64(
        input: cudf_table_view_handle,
        seed: u64,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    // -- Join ----------------------------------------------------------------

    pub fn cudf_inner_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: i32,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_inner_join_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: i32,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_left_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: i32,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_left_join_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: i32,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_full_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: i32,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_full_join_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: i32,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Replace -------------------------------------------------------------

    pub fn cudf_replace_nulls_scalar(
        input: cudf_column_view_handle,
        replacement: cudf_scalar_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_replace_nulls_scalar_ex(
        input: cudf_column_view_handle,
        replacement: cudf_scalar_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_replace_nulls_column(
        input: cudf_column_view_handle,
        replacement: cudf_column_view_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_replace_nulls_column_ex(
        input: cudf_column_view_handle,
        replacement: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Search --------------------------------------------------------------

    pub fn cudf_contains(
        haystack: cudf_column_view_handle,
        needles: cudf_column_view_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    // -- Copying (gather) ----------------------------------------------------

    pub fn cudf_gather(
        input: cudf_table_view_handle,
        gather_map: cudf_column_view_handle,
        bounds_policy: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_gather_ex(
        input: cudf_table_view_handle,
        gather_map: cudf_column_view_handle,
        bounds_policy: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Stream Compaction ---------------------------------------------------

    pub fn cudf_drop_nulls(
        input: cudf_table_view_handle,
        keys: *const i32,
        num_keys: i32,
        keep_threshold: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_distinct(
        input: cudf_table_view_handle,
        keys: *const i32,
        num_keys: i32,
        keep: i32,
        null_equality: i32,
        nan_equality: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    // -- Test observation hooks -----------------------------------------------
    //
    // These functions are defined in `native/src/execution_context.cpp` and
    // allow integration tests to verify that explicit CUDA streams and memory
    // resources are correctly propagated through `_ex` FFI entry points.
    //
    // They are always compiled into `libcudf_rust` (no `#ifdef` guard) but are
    // intended only for test use.  They are unconditionally declared here so
    // that downstream crates (e.g. `cudf-rs` integration tests) can import
    // them from `cudf_sys` instead of re-declaring them locally.

    /// Reset the observation state so the next `_ex` call's context is visible.
    pub fn cudf_test_exec_reset_observation();

    /// Force the next `cudf_table_release_columns` handle-array allocation to
    /// fail for regression testing.
    pub fn cudf_test_fail_next_table_release_columns_handle_array_alloc();

    /// Return the raw `uintptr_t` value of the last stream passed to an `_ex`
    /// shim.
    pub fn cudf_test_exec_last_stream_value(out: *mut usize) -> cudf_status_t;

    /// Return the raw `uintptr_t` value of the last stream that was
    /// synchronized by an `_ex` shim (host-materializing ops).
    pub fn cudf_test_exec_last_sync_stream_value(out: *mut usize) -> cudf_status_t;

    /// Return the raw pointer of the last memory resource passed to an `_ex`
    /// shim.
    pub fn cudf_test_exec_last_mr_raw(out: *mut *mut c_void) -> cudf_status_t;

    /// Return the raw `uintptr_t` value of the process-global default stream.
    pub fn cudf_test_default_stream_value(out: *mut usize) -> cudf_status_t;

    /// Convert a `cudf_stream_handle` to its raw `uintptr_t` value.
    pub fn cudf_test_stream_handle_value(handle: cudf_stream_handle, out: *mut usize) -> cudf_status_t;

    /// Return the raw pointer of the process-global current device memory
    /// resource.
    pub fn cudf_test_current_device_resource_raw(out: *mut *mut c_void) -> cudf_status_t;

    /// Create a statistics-wrapped memory resource for allocator-sensitive
    /// integration tests.
    pub fn cudf_test_memory_resource_create_stats(out: *mut cudf_memory_resource_handle) -> cudf_status_t;

    /// Read the cumulative allocation count from a statistics memory resource.
    pub fn cudf_test_memory_resource_allocation_count(
        handle: cudf_memory_resource_handle,
        out: *mut i64,
    ) -> cudf_status_t;

    /// Install the memory resource owned by `handle` as the process-global
    /// current device resource for tests.
    pub fn cudf_test_memory_resource_set_current(handle: cudf_memory_resource_handle) -> cudf_status_t;

    /// Install a previously observed raw device resource pointer as the
    /// process-global current device resource for tests.
    pub fn cudf_test_memory_resource_set_current_raw(raw_mr: *mut c_void) -> cudf_status_t;

    /// Throw a non-I/O `std::runtime_error` from the native shim for error
    /// classification tests.
    pub fn cudf_test_throw_runtime_error() -> cudf_status_t;

    /// Throw `std::bad_alloc` from the native shim for OOM classification
    /// tests.
    pub fn cudf_test_throw_bad_alloc() -> cudf_status_t;
}
