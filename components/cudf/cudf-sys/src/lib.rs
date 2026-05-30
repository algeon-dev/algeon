//! Low-level FFI bindings to the cuDF C shim library (`libcudf_rust`).
//!
//! This crate provides raw `extern "C"` function declarations matching the
//! C shim layer that wraps cuDF's C++ API. It is not intended to be used
//! directly — use the `cudf` crate for a safe, idiomatic Rust API.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_void};

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
pub type cudf_stream_handle = *mut c_void;
pub type cudf_memory_resource_handle = *mut c_void;

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

    pub fn cudf_column_copy(src: cudf_column_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_column_copy_from_view(src: cudf_column_view_handle, out: *mut cudf_column_handle) -> cudf_status_t;

    pub fn cudf_column_destroy(handle: cudf_column_handle);

    // -- ColumnView (non-owning) --------------------------------------------

    pub fn cudf_column_view(handle: cudf_column_handle, out: *mut cudf_column_view_handle) -> cudf_status_t;

    pub fn cudf_column_view_destroy(handle: cudf_column_view_handle);

    pub fn cudf_column_view_type(handle: cudf_column_view_handle, out: *mut cudf_type_id_t) -> cudf_status_t;

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

    pub fn cudf_column_view_null_mask_to_host(
        handle: cudf_column_view_handle,
        host_buf: *mut u8,
        buf_size: usize,
    ) -> cudf_status_t;

    pub fn cudf_column_create_from_device_data(
        type_id: cudf_type_id_t,
        device_data: *const c_void,
        size: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_create_strings(
        strings: *const *const c_char,
        lengths: *const i32,
        num_strings: i32,
        null_mask: *const u8,
        null_count: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_view_strings_to_host(
        handle: cudf_column_view_handle,
        offsets_buf: *mut i32,
        offsets_buf_size: usize,
        chars_buf: *mut c_char,
        chars_buf_size: usize,
        total_chars_bytes: *mut i32,
    ) -> cudf_status_t;

    // -- Table (owning) ------------------------------------------------------

    pub fn cudf_table_create(
        columns: *mut cudf_column_handle,
        num_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_table_copy(src: cudf_table_handle, out: *mut cudf_table_handle) -> cudf_status_t;

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

    /// Release all columns from a table, destroying the table.
    ///
    /// On success, `table` is consumed and must not be used again.
    /// `out_columns` receives a heap-allocated array of `out_count` column
    /// handles. Each must be destroyed with `cudf_column_destroy`, and the
    /// array freed with `cudf_free_column_array`.
    ///
    /// On failure, `table` is NOT consumed.
    pub fn cudf_table_release_columns(
        table: cudf_table_handle,
        out_columns: *mut *mut cudf_column_handle,
        out_count: *mut i32,
    ) -> cudf_status_t;

    // -- Scalar --------------------------------------------------------------

    pub fn cudf_scalar_create_numeric(
        type_id: cudf_type_id_t,
        value: *const c_void,
        is_valid: i32,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    pub fn cudf_scalar_create_string(
        value: *const c_char,
        is_valid: i32,
        out: *mut cudf_scalar_handle,
    ) -> cudf_status_t;

    pub fn cudf_scalar_destroy(handle: cudf_scalar_handle);

    pub fn cudf_scalar_type(handle: cudf_scalar_handle, out: *mut cudf_type_id_t) -> cudf_status_t;

    pub fn cudf_scalar_is_valid(handle: cudf_scalar_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_scalar_value(handle: cudf_scalar_handle, out: *mut c_void) -> cudf_status_t;

    pub fn cudf_scalar_string_value(
        handle: cudf_scalar_handle,
        out: *mut *const c_char,
        len_out: *mut i32,
    ) -> cudf_status_t;

    // -- Stream --------------------------------------------------------------

    pub fn cudf_stream_create(out: *mut cudf_stream_handle) -> cudf_status_t;
    pub fn cudf_stream_destroy(handle: cudf_stream_handle);
    pub fn cudf_stream_synchronize(handle: cudf_stream_handle) -> cudf_status_t;
    pub fn cudf_stream_is_ready(handle: cudf_stream_handle, out: *mut i32) -> cudf_status_t;

    // -- Memory Resource -----------------------------------------------------

    pub fn cudf_memory_resource_create_cuda(out: *mut cudf_memory_resource_handle) -> cudf_status_t;

    pub fn cudf_memory_resource_create_pool(
        initial_pool_size: usize,
        maximum_pool_size: usize,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

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

    pub fn cudf_io_free_column_names(names: *mut *mut c_char, num_names: i32);

    pub fn cudf_io_write_parquet(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
    ) -> cudf_status_t;

    // -- I/O: CSV ------------------------------------------------------------

    pub fn cudf_io_read_csv(
        filepath: *const c_char,
        delimiter: c_char,
        has_header: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_io_write_csv(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        delimiter: c_char,
        include_header: i32,
    ) -> cudf_status_t;

    // -- Sorting -------------------------------------------------------------

    pub fn cudf_sort(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_stable_sort(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_sort_by_key(
        values: cudf_table_view_handle,
        keys: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_key_columns: i32,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_sorted_order(
        input: cudf_table_view_handle,
        column_order: *const i32,
        null_precedence: *const i32,
        num_keys: i32,
        out: *mut cudf_column_handle,
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

    // -- GroupBy --------------------------------------------------------------

    pub fn cudf_aggregation_create(kind: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_count(null_policy: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_std(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_create_variance(ddof: i32, out: *mut *mut c_void) -> cudf_status_t;

    pub fn cudf_aggregation_destroy(handle: *mut c_void);

    pub fn cudf_groupby_aggregate(
        keys: cudf_table_view_handle,
        values: *const cudf_column_view_handle,
        aggregations: *const *mut c_void,
        agg_counts: *const i32,
        num_requests: i32,
        keys_null_policy: i32,
        out_keys: *mut cudf_table_handle,
        out_values: *mut *mut cudf_column_handle,
        out_total_columns: *mut i32,
    ) -> cudf_status_t;

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

    pub fn cudf_concatenate_tables(
        tables: *const cudf_table_view_handle,
        num_tables: i32,
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

    pub fn cudf_dictionary_match_dictionaries(
        dict_columns: *const cudf_column_view_handle,
        num_columns: i32,
        out: *mut cudf_column_handle,
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

    pub fn cudf_left_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: i32,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
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

    // -- Replace -------------------------------------------------------------

    pub fn cudf_replace_nulls_scalar(
        input: cudf_column_view_handle,
        replacement: cudf_scalar_handle,
        out: *mut cudf_column_handle,
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
}
