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
// Sorting enums (mirrors cudf::order, cudf::null_order)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_order_t {
    CUDF_ORDER_ASCENDING = 0,
    CUDF_ORDER_DESCENDING = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_null_order_t {
    CUDF_NULL_ORDER_AFTER = 0,
    CUDF_NULL_ORDER_BEFORE = 1,
}

// ---------------------------------------------------------------------------
// Groupby enums (mirrors cudf::aggregation::Kind, cudf::null_policy)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_aggregation_kind_t {
    CUDF_AGG_SUM = 0,
    CUDF_AGG_MIN = 1,
    CUDF_AGG_MAX = 2,
    CUDF_AGG_COUNT_VALID = 3,
    CUDF_AGG_COUNT_ALL = 4,
    CUDF_AGG_MEAN = 5,
    CUDF_AGG_MEDIAN = 6,
    CUDF_AGG_NUNIQUE = 7,
    CUDF_AGG_STD = 8,
    CUDF_AGG_VARIANCE = 9,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_null_policy_t {
    CUDF_NULL_POLICY_EXCLUDE = 0,
    CUDF_NULL_POLICY_INCLUDE = 1,
}

// ---------------------------------------------------------------------------
// Join enums (mirrors cudf::null_equality)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_null_equality_t {
    CUDF_NULL_EQUALITY_EQUAL = 0,
    CUDF_NULL_EQUALITY_UNEQUAL = 1,
}

// ---------------------------------------------------------------------------
// Unary enums (mirrors cudf::unary_operator)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_unary_operator_t {
    CUDF_UNARY_SIN = 0,
    CUDF_UNARY_COS = 1,
    CUDF_UNARY_TAN = 2,
    CUDF_UNARY_ARCSIN = 3,
    CUDF_UNARY_ARCCOS = 4,
    CUDF_UNARY_ARCTAN = 5,
    CUDF_UNARY_SINH = 6,
    CUDF_UNARY_COSH = 7,
    CUDF_UNARY_TANH = 8,
    CUDF_UNARY_ARCSINH = 9,
    CUDF_UNARY_ARCCOSH = 10,
    CUDF_UNARY_ARCTANH = 11,
    CUDF_UNARY_EXP = 12,
    CUDF_UNARY_LOG = 13,
    CUDF_UNARY_SQRT = 14,
    CUDF_UNARY_CBRT = 15,
    CUDF_UNARY_CEIL = 16,
    CUDF_UNARY_FLOOR = 17,
    CUDF_UNARY_ABS = 18,
    CUDF_UNARY_RINT = 19,
    CUDF_UNARY_BIT_COUNT = 20,
    CUDF_UNARY_BIT_INVERT = 21,
    CUDF_UNARY_NOT = 22,
    CUDF_UNARY_NEGATE = 23,
}

// ---------------------------------------------------------------------------
// Binary enums (mirrors cudf::binary_operator)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_binary_operator_t {
    CUDF_BINARY_ADD = 0,
    CUDF_BINARY_SUB = 1,
    CUDF_BINARY_MUL = 2,
    CUDF_BINARY_DIV = 3,
    CUDF_BINARY_TRUE_DIV = 4,
    CUDF_BINARY_FLOOR_DIV = 5,
    CUDF_BINARY_MOD = 6,
    CUDF_BINARY_PMOD = 7,
    CUDF_BINARY_PYMOD = 8,
    CUDF_BINARY_POW = 9,
    CUDF_BINARY_INT_POW = 10,
    CUDF_BINARY_LOG_BASE = 11,
    CUDF_BINARY_ATAN2 = 12,
    CUDF_BINARY_SHIFT_LEFT = 13,
    CUDF_BINARY_SHIFT_RIGHT = 14,
    CUDF_BINARY_SHIFT_RIGHT_UNSIGNED = 15,
    CUDF_BINARY_BITWISE_AND = 16,
    CUDF_BINARY_BITWISE_OR = 17,
    CUDF_BINARY_BITWISE_XOR = 18,
    CUDF_BINARY_LOGICAL_AND = 19,
    CUDF_BINARY_LOGICAL_OR = 20,
    CUDF_BINARY_EQUAL = 21,
    CUDF_BINARY_NOT_EQUAL = 22,
    CUDF_BINARY_LESS = 23,
    CUDF_BINARY_GREATER = 24,
    CUDF_BINARY_LESS_EQUAL = 25,
    CUDF_BINARY_GREATER_EQUAL = 26,
    CUDF_BINARY_NULL_EQUALS = 27,
    CUDF_BINARY_NULL_NOT_EQUALS = 28,
    CUDF_BINARY_NULL_MAX = 29,
    CUDF_BINARY_NULL_MIN = 30,
}

// ---------------------------------------------------------------------------
// Reduction enums (mirrors cudf::scan_type, reduction aggregation kind)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_reduce_aggregation_kind_t {
    CUDF_REDUCE_AGG_SUM = 0,
    CUDF_REDUCE_AGG_MIN = 1,
    CUDF_REDUCE_AGG_MAX = 2,
    CUDF_REDUCE_AGG_PRODUCT = 3,
    CUDF_REDUCE_AGG_MEAN = 4,
    CUDF_REDUCE_AGG_MEDIAN = 5,
    CUDF_REDUCE_AGG_ANY = 6,
    CUDF_REDUCE_AGG_ALL = 7,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_scan_type_t {
    CUDF_SCAN_INCLUSIVE = 0,
    CUDF_SCAN_EXCLUSIVE = 1,
}

// ---------------------------------------------------------------------------
// Copying enums (mirrors cudf::out_of_bounds_policy)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_out_of_bounds_policy_t {
    CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY = 0,
    CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK = 1,
}

// ---------------------------------------------------------------------------
// Stream compaction enums
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_duplicate_keep_option_t {
    CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY = 0,
    CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST = 1,
    CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST = 2,
    CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE = 3,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_stream_compaction_null_equality_t {
    CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL = 0,
    CUDF_STREAM_COMPACTION_NULL_EQUALITY_UNEQUAL = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_stream_compaction_nan_equality_t {
    CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL = 0,
    CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL = 1,
}

// ---------------------------------------------------------------------------
// String enums (mirrors cudf::strings::side_type)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_strip_side_t {
    CUDF_STRIP_LEFT = 0,
    CUDF_STRIP_RIGHT = 1,
    CUDF_STRIP_BOTH = 2,
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
pub type cudf_event_handle = *mut c_void;
pub type cudf_memory_resource_handle = *mut c_void;

// ---------------------------------------------------------------------------
// Execution options and typed handle wrappers
// ---------------------------------------------------------------------------

pub const CUDF_EXEC_OPTIONS_RESERVED_SLOTS: usize = 4;

/// Bit 0 of `flags`: when set, `device` was explicitly provided.
pub const CUDF_EXEC_FLAG_DEVICE_EXPLICIT: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_exec_options_t {
    pub struct_size: usize,
    pub device: i32,
    pub flags: u32,
    pub stream: cudf_stream_handle,
    pub mr: cudf_memory_resource_handle,
    pub reserved: [u64; CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
}

impl cudf_exec_options_t {
    pub const fn defaults() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            device: -1,
            flags: 0,
            stream: std::ptr::null_mut(),
            mr: std::ptr::null_mut(),
            reserved: [0; CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
        }
    }
}

impl Default for cudf_exec_options_t {
    fn default() -> Self {
        Self::defaults()
    }
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_interop_placement_kind_t {
    CUDF_INTEROP_PLACEMENT_HOST = 0,
    CUDF_INTEROP_PLACEMENT_CUDA_DEVICE = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_interop_validity_kind_t {
    CUDF_INTEROP_VALIDITY_NONE = 0,
    CUDF_INTEROP_VALIDITY_MASK = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_interop_layout_kind_t {
    CUDF_INTEROP_LAYOUT_FIXED_WIDTH = 0,
    CUDF_INTEROP_LAYOUT_BINARY_LIKE = 1,
    CUDF_INTEROP_LAYOUT_LIST = 2,
    CUDF_INTEROP_LAYOUT_STRUCT = 3,
    CUDF_INTEROP_LAYOUT_DICTIONARY = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_interop_buffer_binding_t {
    pub buffer_id: u32,
    pub data: *const c_void,
    pub byte_len: usize,
    pub placement_kind: cudf_interop_placement_kind_t,
    pub device: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_interop_semantic_column_t {
    pub column_id: u32,
    pub type_id: cudf_type_id_t,
    pub scale: i32,
    pub len: i32,
    pub offset: i32,
    pub validity_kind: cudf_interop_validity_kind_t,
    pub validity_buffer_id: u32,
    pub null_count: i32,
    pub layout_kind: cudf_interop_layout_kind_t,
    pub values_buffer_id: u32,
    pub offsets_buffer_id: u32,
    pub data_buffer_id: u32,
    pub indices_buffer_id: u32,
    pub child_column_id: u32,
    pub child_column_ids: *const u32,
    pub num_child_column_ids: i32,
    pub keys_column_id: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_interop_semantic_table_t {
    pub num_rows: i32,
    pub column_ids: *const u32,
    pub num_columns: i32,
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

    /// Borrow the underlying raw CUDA stream pointer.
    ///
    /// The returned pointer does not transfer ownership and is valid only while
    /// this owning handle remains alive.
    ///
    /// Do not wrap the returned pointer in another owning RAII handle. Mixing
    /// multiple owners for the same raw stream, including higher-level cudf
    /// wrappers, can double-destroy the underlying CUDA object.
    pub fn raw_cuda_stream(&self) -> Result<*mut c_void, cudf_status_t> {
        let mut out = std::ptr::null_mut();
        status_to_result(unsafe { cudf_stream_raw(self.handle, &mut out) })?;
        Ok(out)
    }

    pub fn wait_event(&self, event: &Event) -> Result<(), cudf_status_t> {
        status_to_result(unsafe { cudf_stream_wait_event(self.handle, event.handle) })
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
pub struct Event {
    handle: cudf_event_handle,
}

impl Event {
    pub fn new() -> Result<Self, cudf_status_t> {
        let mut handle = std::ptr::null_mut();
        status_to_result(unsafe { cudf_event_create(&mut handle) })?;
        Ok(Self { handle })
    }

    pub fn as_raw(&self) -> cudf_event_handle {
        self.handle
    }

    pub fn record(&self, stream: &Stream) -> Result<(), cudf_status_t> {
        status_to_result(unsafe { cudf_event_record(self.handle, stream.handle) })
    }

    pub fn synchronize(&self) -> Result<(), cudf_status_t> {
        status_to_result(unsafe { cudf_event_synchronize(self.handle) })
    }

    pub fn is_ready(&self) -> Result<bool, cudf_status_t> {
        let mut out = 0;
        status_to_result(unsafe { cudf_event_is_ready(self.handle, &mut out) })?;
        Ok(out != 0)
    }

    pub fn raw_cuda_event(&self) -> Result<*mut c_void, cudf_status_t> {
        let mut out = std::ptr::null_mut();
        // The returned pointer is borrowed-only. Creating a second owning
        // wrapper for the same event handle can double-destroy it.
        status_to_result(unsafe { cudf_event_raw(self.handle, &mut out) })?;
        Ok(out)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cudf_event_destroy(self.handle) };
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

    /// Borrow the underlying raw `device_memory_resource*`.
    ///
    /// The returned pointer does not transfer ownership and is valid only while
    /// this owning handle remains alive.
    ///
    /// Do not promote this borrowed pointer into another owning wrapper unless
    /// ownership has been explicitly transferred. Two owners for the same raw
    /// memory-resource handle will double-destroy it.
    pub fn raw_device_resource(&self) -> Result<*mut c_void, cudf_status_t> {
        let mut out = std::ptr::null_mut();
        status_to_result(unsafe { cudf_memory_resource_raw(self.handle, &mut out) })?;
        Ok(out)
    }

    pub fn test_raw_device_resource(&self) -> Result<*mut c_void, cudf_status_t> {
        self.raw_device_resource()
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
///
/// # Safety
/// `raw_mr` must be a valid pointer to an `rmm::device_memory_resource` that
/// outlives the test scope. Passing a dangling or mistyped pointer is UB.
pub unsafe fn test_set_current_device_resource_raw(raw_mr: *mut c_void) -> Result<(), cudf_status_t> {
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

    /// Create a fixed-width column by copying from borrowed host buffers.
    ///
    /// This function synchronizes cuDF's default stream before return on
    /// success so `data` and `null_mask` are no longer in use.
    pub fn cudf_column_create_numeric(
        type_id: cudf_type_id_t,
        data: *const c_void,
        size: i32,
        null_mask: *const u8,
        null_count: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a fixed-width column by copying from borrowed host buffers with
    /// explicit execution options.
    ///
    /// The selected stream is synchronized before return on success so `data`
    /// and `null_mask` are no longer in use.
    /// When `options->mr` is non-null, it must outlive the returned column.
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

    pub fn cudf_column_copy_from_borrowed_view(
        type_id: cudf_type_id_t,
        scale: i32,
        size: i32,
        offset: i32,
        data: *const c_void,
        null_mask: *const c_void,
        null_count: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_copy_from_borrowed_view_ex(
        type_id: cudf_type_id_t,
        scale: i32,
        size: i32,
        offset: i32,
        data: *const c_void,
        null_mask: *const c_void,
        null_count: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_copy_from_prepared_device_export(
        root_column_id: u32,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_copy_from_prepared_device_export_ex(
        root_column_id: u32,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_copy_from_prepared_host_export(
        root_column_id: u32,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_copy_from_prepared_host_export_ex(
        root_column_id: u32,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_payload_size(handle: cudf_column_handle, out: *mut usize) -> cudf_status_t;

    pub fn cudf_column_payload_size_ex(
        handle: cudf_column_handle,
        options: *const cudf_exec_options_t,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_column_device_allocated_size(handle: cudf_column_handle, out: *mut usize) -> cudf_status_t;

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

    pub fn cudf_column_view_list_sliced_child(
        handle: cudf_column_view_handle,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_view_list_sliced_child_ex(
        handle: cudf_column_view_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_view_data_ptr(handle: cudf_column_view_handle, out: *mut *const c_void) -> cudf_status_t;

    pub fn cudf_column_view_string_chars_ptr_and_size(
        handle: cudf_column_view_handle,
        out_ptr: *mut *const c_void,
        out_size: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_column_view_string_chars_ptr_and_size_ex(
        handle: cudf_column_view_handle,
        options: *const cudf_exec_options_t,
        out_ptr: *mut *const c_void,
        out_size: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_column_view_null_mask_ptr(handle: cudf_column_view_handle, out: *mut *const c_void) -> cudf_status_t;

    pub fn cudf_column_view_offset(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_column_view_slice(
        handle: cudf_column_view_handle,
        offset: i32,
        length: i32,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

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
    /// `host_buf` is fully populated on success. For sliced views, the
    /// exported host bitmask is normalized to the logical slice so the view's
    /// first row maps to bit 0.
    pub fn cudf_column_view_null_mask_to_host(
        handle: cudf_column_view_handle,
        host_buf: *mut u8,
        buf_size: usize,
    ) -> cudf_status_t;

    /// Materialize a column null mask into a host buffer with explicit
    /// execution options.
    ///
    /// On success, the host buffer is fully populated before the function
    /// returns. For sliced views, the exported host bitmask is normalized to
    /// the logical slice so the view's first row maps to bit 0.
    pub fn cudf_column_view_null_mask_to_host_ex(
        handle: cudf_column_view_handle,
        host_buf: *mut u8,
        buf_size: usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Create a column by copying from a borrowed device pointer.
    ///
    /// This function synchronizes cuDF's default stream before return so
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
    ///
    /// Passing null for `options`, or leaving `options->stream` /
    /// `options->mr` null, uses cuDF's default stream and current device
    /// memory resource. When `options->mr` is non-null, it must outlive the
    /// returned column.
    pub fn cudf_column_from_scalar_ex(
        scalar: cudf_scalar_handle,
        size: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Create a STRING column by copying from borrowed host buffers.
    ///
    /// This function synchronizes cuDF's default stream before return on
    /// success so the borrowed host buffers are no longer in use.
    pub fn cudf_column_create_strings(
        strings: *const *const c_char,
        lengths: *const i32,
        num_strings: i32,
        null_mask: *const u8,
        null_count: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a STRING column by copying from borrowed host buffers with
    /// explicit execution options.
    ///
    /// The selected stream is synchronized before return on success so the
    /// borrowed host buffers are no longer in use.
    /// When `options->mr` is non-null, it must outlive the returned column.
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
    /// Empty STRING columns still require `offsets_buf`; they write exactly
    /// one zero offset (`offsets_buf[0] = 0`) and report
    /// `*total_chars_bytes = 0`. In that case `chars_buf` is ignored because
    /// there are no chars to copy.
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

    /// Materialize a STRING column into caller-provided host buffers with
    /// explicit execution options.
    ///
    /// Empty STRING columns still require `offsets_buf`; they write exactly
    /// one zero offset (`offsets_buf[0] = 0`) and report
    /// `*total_chars_bytes = 0`. In that case `chars_buf` is ignored because
    /// there are no chars to copy.
    ///
    /// The selected stream is synchronized before return whenever offsets or
    /// chars are materialized into the caller's host buffers. `options->mr` is
    /// ignored because this helper does not allocate output device storage.
    pub fn cudf_column_view_strings_to_host_ex(
        handle: cudf_column_view_handle,
        offsets_buf: *mut i32,
        offsets_buf_size: usize,
        chars_buf: *mut c_char,
        chars_buf_size: usize,
        total_chars_bytes: *mut i32,
        options: *const cudf_exec_options_t,
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

    pub fn cudf_table_copy_from_prepared_device_export(
        table: *const cudf_interop_semantic_table_t,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy_from_prepared_device_export_ex(
        table: *const cudf_interop_semantic_table_t,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy_from_prepared_host_export(
        table: *const cudf_interop_semantic_table_t,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy_from_prepared_host_export_ex(
        table: *const cudf_interop_semantic_table_t,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_payload_size(handle: cudf_table_handle, out: *mut usize) -> cudf_status_t;

    pub fn cudf_table_payload_size_ex(
        handle: cudf_table_handle,
        options: *const cudf_exec_options_t,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_table_device_allocated_size(handle: cudf_table_handle, out: *mut usize) -> cudf_status_t;

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

    /// Materialize a numeric scalar value with explicit execution options.
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

    // -- Device ---------------------------------------------------------------

    /// Query the CUDA device ordinal for the calling thread.
    pub fn cudf_cuda_get_device(device: *mut i32) -> cudf_status_t;

    /// Set the CUDA device ordinal for the calling thread.
    pub fn cudf_cuda_set_device(device: i32) -> cudf_status_t;

    /// Query the number of CUDA-capable devices visible to the process.
    pub fn cudf_cuda_get_device_count(count: *mut i32) -> cudf_status_t;

    /// Resolve the CUDA device ordinal that owns a device pointer.
    pub fn cudf_cuda_pointer_get_device(ptr: *const c_void, device: *mut i32) -> cudf_status_t;

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

    /// Borrow the underlying raw CUDA stream pointer from `handle`.
    ///
    /// This does not transfer ownership. The returned pointer is valid only
    /// while `handle` remains alive.
    pub fn cudf_stream_raw(handle: cudf_stream_handle, out: *mut *mut c_void) -> cudf_status_t;

    // -- Event ---------------------------------------------------------------

    /// Create an owned CUDA event handle.
    pub fn cudf_event_create(out: *mut cudf_event_handle) -> cudf_status_t;

    /// Destroy a CUDA event created by [`cudf_event_create`].
    pub fn cudf_event_destroy(handle: cudf_event_handle);

    /// Record an event on a CUDA stream handle.
    pub fn cudf_event_record(event: cudf_event_handle, stream: cudf_stream_handle) -> cudf_status_t;

    /// Block until a recorded event completes.
    pub fn cudf_event_synchronize(event: cudf_event_handle) -> cudf_status_t;

    /// Report whether a recorded event has completed.
    pub fn cudf_event_is_ready(event: cudf_event_handle, out: *mut i32) -> cudf_status_t;

    /// Make `stream` wait for `event` before future work proceeds.
    pub fn cudf_stream_wait_event(stream: cudf_stream_handle, event: cudf_event_handle) -> cudf_status_t;

    /// Record a borrowed raw CUDA event on a borrowed raw CUDA stream.
    pub fn cudf_raw_cuda_event_record(raw_event: *mut c_void, raw_stream: *mut c_void) -> cudf_status_t;

    /// Make a borrowed raw CUDA stream wait on a borrowed raw CUDA event.
    pub fn cudf_raw_cuda_stream_wait_event(raw_stream: *mut c_void, raw_event: *mut c_void) -> cudf_status_t;

    /// Borrow the underlying raw CUDA event pointer from `handle`.
    ///
    /// This does not transfer ownership. The returned pointer is valid only
    /// while `handle` remains alive.
    pub fn cudf_event_raw(handle: cudf_event_handle, out: *mut *mut c_void) -> cudf_status_t;

    /// Reset CUDA event observation hooks used by integration tests.
    pub fn cudf_test_event_reset_observation();

    /// Read the number of observed `cudaEventSynchronize` calls.
    pub fn cudf_test_event_synchronize_call_count(out: *mut i32) -> cudf_status_t;

    /// Read the number of observed `cudaStreamWaitEvent` calls.
    pub fn cudf_test_event_stream_wait_call_count(out: *mut i32) -> cudf_status_t;

    /// Read the number of observed `cudaStreamWaitEvent` calls for one raw stream.
    pub fn cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw: usize, out: *mut i32) -> cudf_status_t;

    /// Read the number of observed `cudaStreamWaitEvent` calls for one raw event.
    pub fn cudf_test_event_stream_wait_call_count_for_event_raw(event_raw: usize, out: *mut i32) -> cudf_status_t;

    /// Read the last raw CUDA stream pointer passed to `cudaStreamWaitEvent`.
    pub fn cudf_test_event_last_wait_stream_raw(out: *mut usize) -> cudf_status_t;

    /// Read the last raw CUDA event pointer passed to `cudaStreamWaitEvent`.
    pub fn cudf_test_event_last_wait_event_raw(out: *mut usize) -> cudf_status_t;

    /// Read the number of observed raw `cudaEventRecord` calls.
    pub fn cudf_test_event_raw_record_call_count(out: *mut i32) -> cudf_status_t;

    /// Read the number of observed raw `cudaEventRecord` calls for one raw stream.
    pub fn cudf_test_event_raw_record_call_count_for_stream_raw(stream_raw: usize, out: *mut i32) -> cudf_status_t;

    /// Read the last raw CUDA stream pointer passed to raw `cudaEventRecord`.
    pub fn cudf_test_event_last_raw_record_stream_raw(out: *mut usize) -> cudf_status_t;

    /// Read the last raw CUDA event pointer passed to raw `cudaEventRecord`.
    pub fn cudf_test_event_last_raw_record_event_raw(out: *mut usize) -> cudf_status_t;

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

    /// Borrow the underlying raw `device_memory_resource*` from `handle`.
    ///
    /// This does not transfer ownership. The returned pointer is valid only
    /// while `handle` remains alive.
    pub fn cudf_memory_resource_raw(handle: cudf_memory_resource_handle, out: *mut *mut c_void) -> cudf_status_t;

    // -- I/O: Parquet --------------------------------------------------------

    pub fn cudf_io_read_parquet(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Read Parquet with explicit execution context.
    ///
    /// Passing null for `options`, or leaving `options->stream` /
    /// `options->mr` null, uses cuDF's default stream and current device
    /// memory resource. When `options->mr` is non-null, it must outlive the
    /// returned table.
    pub fn cudf_io_read_parquet_ex(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_io_read_parquet_with_metadata(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
    ) -> cudf_status_t;

    /// Read Parquet with metadata and explicit execution context.
    ///
    /// Passing null for `options`, or leaving `options->stream` /
    /// `options->mr` null, uses cuDF's default stream and current device
    /// memory resource. When `options->mr` is non-null, it must outlive the
    /// returned table.
    pub fn cudf_io_read_parquet_with_metadata_ex(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
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
        row_group_size_rows: i32,
    ) -> cudf_status_t;

    /// Write Parquet with explicit execution context.
    /// NOTE: libcudf's write_parquet only accepts stream, not mr.
    /// The mr field in options is ignored for writes.
    pub fn cudf_io_write_parquet_ex(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        row_group_size_rows: i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- I/O: CSV ------------------------------------------------------------

    pub fn cudf_io_read_csv(
        filepath: *const c_char,
        delimiter: c_char,
        has_header: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Read a CSV file into a table with explicit execution context.
    ///
    /// Passing null for `options`, or leaving `options->stream` /
    /// `options->mr` null, uses cuDF's default stream and current device
    /// memory resource. When `options->mr` is non-null, it must outlive the
    /// returned table.
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

    // -- JSON I/O ------------------------------------------------------------

    pub fn cudf_io_read_json(filepath: *const c_char, lines: i32, out: *mut cudf_table_handle) -> cudf_status_t;
    pub fn cudf_io_read_json_ex(
        filepath: *const c_char,
        lines: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
    pub fn cudf_io_read_json_with_metadata(
        filepath: *const c_char,
        lines: i32,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
    ) -> cudf_status_t;
    pub fn cudf_io_read_json_with_metadata_ex(
        filepath: *const c_char,
        lines: i32,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
    pub fn cudf_io_write_json(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        lines: i32,
        include_nulls: i32,
    ) -> cudf_status_t;
    pub fn cudf_io_write_json_ex(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        lines: i32,
        include_nulls: i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Sorting -------------------------------------------------------------

    pub fn cudf_sort(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Sort with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_sort_ex(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_stable_sort(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Stable sort with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_stable_sort_ex(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_sort_by_key(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Sort by key with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_sort_by_key_ex(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_stable_sort_by_key(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Stable sort by key with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_stable_sort_by_key_ex(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_sorted_order(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Sorted order with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned column.
    pub fn cudf_sorted_order_ex(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_is_sorted(
        input: cudf_table_view_handle,
        column_order: *const cudf_order_t,
        null_precedence: *const cudf_null_order_t,
        num_keys: i32,
        out: *mut i32,
    ) -> cudf_status_t;

    // -- Filter --------------------------------------------------------------

    pub fn cudf_apply_boolean_mask(
        input: cudf_table_view_handle,
        mask: cudf_column_view_handle,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Apply a boolean mask with explicit execution options.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_apply_boolean_mask_ex(
        input: cudf_table_view_handle,
        mask: cudf_column_view_handle,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- GroupBy --------------------------------------------------------------

    pub fn cudf_aggregation_create(kind: cudf_aggregation_kind_t, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_count(null_policy: cudf_null_policy_t, out: *mut *mut c_void) -> cudf_status_t;

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
        keys_null_policy: cudf_null_policy_t,
        out_keys: *mut cudf_table_handle,
        out_values: *mut *mut cudf_column_handle,
        out_total_columns: *mut i32,
    ) -> cudf_status_t;

    /// Groupby aggregate with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned keys
    /// table and every returned result column.
    pub fn cudf_groupby_aggregate_ex(
        keys: cudf_table_view_handle,
        values: *const cudf_column_view_handle,
        aggregations: *const *mut c_void,
        agg_counts: *const i32,
        num_requests: i32,
        num_aggregations: i32,
        keys_null_policy: cudf_null_policy_t,
        out_keys: *mut cudf_table_handle,
        out_values: *mut *mut cudf_column_handle,
        out_total_columns: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Free a heap-allocated array of owning column handles returned by this
    /// shim. Does not destroy the individual columns.
    pub fn cudf_free_column_array(array: *mut cudf_column_handle);

    // -- Unary ---------------------------------------------------------------

    pub fn cudf_unary_operation(
        input: cudf_column_view_handle,
        op: cudf_unary_operator_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_unary_operation_ex(
        input: cudf_column_view_handle,
        op: cudf_unary_operator_t,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_is_null(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_is_null_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_is_valid(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_is_valid_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_cast(
        input: cudf_column_view_handle,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_cast_ex(
        input: cudf_column_view_handle,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_is_supported_cast(
        from_type: cudf_data_type_handle,
        to_type: cudf_data_type_handle,
        out: *mut i32,
    ) -> cudf_status_t;

    pub fn cudf_is_nan(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_is_nan_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_is_not_nan(input: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_is_not_nan_ex(
        input: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Binary --------------------------------------------------------------

    pub fn cudf_binary_operation_vv(
        lhs: cudf_column_view_handle,
        rhs: cudf_column_view_handle,
        op: cudf_binary_operator_t,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_binary_operation_vv_ex(
        lhs: cudf_column_view_handle,
        rhs: cudf_column_view_handle,
        op: cudf_binary_operator_t,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_binary_operation_vs(
        lhs: cudf_column_view_handle,
        rhs: cudf_scalar_handle,
        op: cudf_binary_operator_t,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_binary_operation_vs_ex(
        lhs: cudf_column_view_handle,
        rhs: cudf_scalar_handle,
        op: cudf_binary_operator_t,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_binary_operation_sv(
        lhs: cudf_scalar_handle,
        rhs: cudf_column_view_handle,
        op: cudf_binary_operator_t,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_binary_operation_sv_ex(
        lhs: cudf_scalar_handle,
        rhs: cudf_column_view_handle,
        op: cudf_binary_operator_t,
        output_type: cudf_data_type_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Reduction -----------------------------------------------------------

    pub fn cudf_reduce_aggregation_create(kind: cudf_reduce_aggregation_kind_t, out: *mut *mut c_void)
    -> cudf_status_t;

    pub fn cudf_reduce_aggregation_create_variance(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_reduce_aggregation_create_std(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_reduce_aggregation_destroy(handle: *mut c_void);

    pub fn cudf_scan_aggregation_create(kind: cudf_reduce_aggregation_kind_t, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_scan_aggregation_destroy(handle: *mut c_void);

    pub fn cudf_reduce(
        input: cudf_column_view_handle,
        agg: *mut c_void,
        output_type: cudf_data_type_handle,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned scalar.
    pub fn cudf_reduce_ex(
        input: cudf_column_view_handle,
        agg: *mut c_void,
        output_type: cudf_data_type_handle,
        out: *mut cudf_scalar_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_scan(
        input: cudf_column_view_handle,
        agg: *mut c_void,
        scan_type: cudf_scan_type_t,
        null_policy: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned column.
    pub fn cudf_scan_ex(
        input: cudf_column_view_handle,
        agg: *mut c_void,
        scan_type: cudf_scan_type_t,
        null_policy: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_minmax(
        input: cudf_column_view_handle,
        out_min: *mut cudf_scalar_handle,
        out_max: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    /// When `options->mr` is non-NULL, it must outlive the returned scalars.
    pub fn cudf_minmax_ex(
        input: cudf_column_view_handle,
        out_min: *mut cudf_scalar_handle,
        out_max: *mut cudf_scalar_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Concatenate ---------------------------------------------------------

    pub fn cudf_concatenate_columns(
        columns: *const cudf_column_view_handle,
        num_columns: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Concatenate multiple columns into one with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned column.
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

    /// Concatenate multiple tables into one with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
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

    pub fn cudf_dictionary_encode_ex(
        input: cudf_column_view_handle,
        indices_type_id: cudf_type_id_t,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_dictionary_decode(dict_column: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_dictionary_decode_ex(
        dict_column: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

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

    pub fn cudf_dictionary_set_keys_ex(
        dict_column: cudf_column_view_handle,
        keys: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
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

    pub fn cudf_dictionary_match_dictionaries_ex(
        dict_columns: *const cudf_column_view_handle,
        num_columns: i32,
        out_columns: *mut *mut cudf_column_handle,
        out_count: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Hashing -------------------------------------------------------------

    pub fn cudf_hashing_murmurhash3_x86_32(
        input: cudf_table_view_handle,
        seed: u32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hashing_murmurhash3_x86_32_ex(
        input: cudf_table_view_handle,
        seed: u32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hashing_xxhash_64(
        input: cudf_table_view_handle,
        seed: u64,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hashing_xxhash_64_ex(
        input: cudf_table_view_handle,
        seed: u64,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    // -- Join ----------------------------------------------------------------

    pub fn cudf_inner_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Inner join with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive both returned tables.
    pub fn cudf_inner_join_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_left_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Left join with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive both returned tables.
    pub fn cudf_left_join_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_full_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Full join with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive both returned tables.
    pub fn cudf_full_join_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
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

    /// Replace nulls from a scalar with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned column.
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

    /// Replace nulls from another column with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned column.
    pub fn cudf_replace_nulls_column_ex(
        input: cudf_column_view_handle,
        replacement: cudf_column_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_replace_nulls_policy(
        input: cudf_column_view_handle,
        policy: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;
    pub fn cudf_replace_nulls_policy_ex(
        input: cudf_column_view_handle,
        policy: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

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

    // -- Copying (gather) ----------------------------------------------------

    pub fn cudf_gather(
        input: cudf_table_view_handle,
        gather_map: cudf_column_view_handle,
        bounds_policy: cudf_out_of_bounds_policy_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Gather rows with explicit execution options.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_gather_ex(
        input: cudf_table_view_handle,
        gather_map: cudf_column_view_handle,
        bounds_policy: cudf_out_of_bounds_policy_t,
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

    pub fn cudf_drop_nulls_ex(
        input: cudf_table_view_handle,
        keys: *const i32,
        num_keys: i32,
        keep_threshold: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_distinct(
        input: cudf_table_view_handle,
        keys: *const i32,
        num_keys: i32,
        keep: cudf_duplicate_keep_option_t,
        null_equality: cudf_stream_compaction_null_equality_t,
        nan_equality: cudf_stream_compaction_nan_equality_t,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_distinct_ex(
        input: cudf_table_view_handle,
        keys: *const i32,
        num_keys: i32,
        keep: cudf_duplicate_keep_option_t,
        null_equality: cudf_stream_compaction_null_equality_t,
        nan_equality: cudf_stream_compaction_nan_equality_t,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
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

    // -- Test helpers --------------------------------------------------------

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

    /// Return the CUDA device ordinal that was current when the last `_ex`
    /// shim resolved execution options, or -1 if no observation was recorded.
    pub fn cudf_test_exec_last_device(out: *mut i32) -> cudf_status_t;

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

    /// Resolve a memory-resource handle to its underlying `device_memory_resource*`.
    pub fn cudf_test_memory_resource_raw(handle: cudf_memory_resource_handle, out: *mut *mut c_void) -> cudf_status_t;

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
