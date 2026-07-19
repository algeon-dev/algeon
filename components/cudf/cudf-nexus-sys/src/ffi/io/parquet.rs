use std::os::raw::{c_char, c_void};

use crate::{
    cudf_ast_expression_handle, cudf_exec_options_t, cudf_status_t, cudf_table_handle, cudf_table_view_handle,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_io_datasource_probe_result_t {
    pub can_open: i32,
    pub supports_device_read: i32,
    pub device_read_preferred: i32,
    pub size_bytes: i64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_io_remote_parquet_profile_t {
    pub aws_region: *const c_char,
    pub aws_access_key_id: *const c_char,
    pub aws_secret_access_key: *const c_char,
    pub aws_session_token: *const c_char,
    pub aws_endpoint_url: *const c_char,
    pub path_style_access: i32,
    pub unsigned_request: i32,
    pub known_size_bytes: i64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_io_remote_parquet_read_evidence_t {
    pub backend: i32,
    pub fallback_reason: i32,
    pub supports_device_read: i32,
    pub device_read_used: i32,
    pub remote_bytes: i64,
    pub remote_range_count: i64,
    pub host_read_bytes: i64,
    pub host_read_count: i64,
    pub device_read_bytes: i64,
    pub device_read_count: i64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_io_parquet_pruning_evidence_t {
    pub input_row_groups: i64,
    pub has_row_groups_after_stats_filter: i32,
    pub row_groups_after_stats_filter: i64,
    pub has_row_groups_after_bloom_filter: i32,
    pub row_groups_after_bloom_filter: i64,
}

pub const CUDF_IO_KVIKIO_COMPAT_MODE_AUTO: i32 = 0;
pub const CUDF_IO_KVIKIO_COMPAT_MODE_ON: i32 = 1;
pub const CUDF_IO_KVIKIO_COMPAT_MODE_OFF: i32 = 2;

unsafe extern "C" {
    // -- I/O: Parquet --------------------------------------------------------

    /// Read Parquet with explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned table.
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

    /// Read Parquet from a host-memory buffer with explicit execution context.
    ///
    /// `host_buffer` is borrowed only for the duration of this call. `options`
    /// must name an explicit valid device and non-null stream/MR.
    pub fn cudf_io_read_parquet_from_host_buffer_ex(
        host_buffer: *const c_void,
        host_buffer_len: usize,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Read Parquet with metadata and explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// `options->mr` must outlive the returned table.
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

    /// Read Parquet with metadata, explicit execution context, and an optional
    /// libcudf AST reader predicate.
    ///
    /// Passing null for `predicate` preserves ordinary unfiltered reader
    /// behavior. `options` must name an explicit valid device and non-null
    /// stream/MR. `options->mr` must outlive the returned table.
    pub fn cudf_io_read_parquet_with_metadata_filter_ex(
        filepath: *const c_char,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        predicate: cudf_ast_expression_handle,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        out_pruning_evidence: *mut cudf_io_parquet_pruning_evidence_t,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Read remote Parquet through an explicit KvikIO-backed datasource with
    /// scoped S3 options and read evidence.
    pub fn cudf_io_read_remote_parquet_with_metadata_filter_ex(
        uri: *const c_char,
        profile: *const cudf_io_remote_parquet_profile_t,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        predicate: cudf_ast_expression_handle,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        out_evidence: *mut cudf_io_remote_parquet_read_evidence_t,
        out_pruning_evidence: *mut cudf_io_parquet_pruning_evidence_t,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Read multiple Parquet files with metadata, explicit execution context,
    /// and an optional libcudf AST reader predicate.
    ///
    /// `filepaths` must contain `num_files` non-null strings. Per-file row
    /// groups are encoded by `row_group_offsets` with length
    /// `num_files + 1`; pass null offsets and zero row groups to read all row
    /// groups from every file.
    pub fn cudf_io_read_parquet_multi_with_metadata_filter_ex(
        filepaths: *const *const c_char,
        num_files: i32,
        columns: *const *const c_char,
        num_columns: i32,
        row_group_offsets: *const i32,
        num_row_group_offsets: i32,
        row_groups: *const i32,
        num_row_groups: i32,
        predicate: cudf_ast_expression_handle,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        out_pruning_evidence: *mut cudf_io_parquet_pruning_evidence_t,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Read multiple Parquet files with optional explicit remote datasource
    /// entries and aggregate remote read evidence.
    pub fn cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex(
        filepaths: *const *const c_char,
        remote_source_flags: *const i32,
        profiles: *const cudf_io_remote_parquet_profile_t,
        num_files: i32,
        columns: *const *const c_char,
        num_columns: i32,
        row_group_offsets: *const i32,
        num_row_group_offsets: i32,
        row_groups: *const i32,
        num_row_groups: i32,
        predicate: cudf_ast_expression_handle,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        out_evidence: *mut cudf_io_remote_parquet_read_evidence_t,
        out_pruning_evidence: *mut cudf_io_parquet_pruning_evidence_t,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Read Parquet from a host-memory buffer with metadata, explicit
    /// execution context, and an optional libcudf AST reader predicate.
    ///
    /// Passing null for `predicate` preserves ordinary unfiltered reader
    /// behavior. `host_buffer` is borrowed only for the duration of this call.
    pub fn cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex(
        host_buffer: *const c_void,
        host_buffer_len: usize,
        columns: *const *const c_char,
        num_columns: i32,
        skip_rows: i64,
        num_rows: i64,
        row_groups: *const i32,
        num_row_groups: i32,
        predicate: cudf_ast_expression_handle,
        out_table: *mut cudf_table_handle,
        out_names: *mut *mut *mut c_char,
        out_num_names: *mut i32,
        out_pruning_evidence: *mut cudf_io_parquet_pruning_evidence_t,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Free a heap-allocated array of host column names returned by this shim.
    ///
    /// When the associated count is zero, the shim returns a null pointer and
    /// this function may be skipped.
    pub fn cudf_io_free_column_names(names: *mut *mut c_char, num_names: i32);

    /// Returns whether libcudf was compiled with CUDF_KVIKIO_REMOTE_IO:
    /// 1 = enabled, 0 = disabled, -1 = unknown/reserved.
    pub fn cudf_io_remote_parquet_kvikio_remote_io_enabled() -> i32;

    pub fn cudf_io_kvikio_set_compat_mode(compat_mode: i32) -> cudf_status_t;

    pub fn cudf_io_kvikio_compat_mode() -> i32;

    pub fn cudf_io_kvikio_drain_bounce_buffer_pools(
        out_cleared_bytes: *mut u64,
        out_active_io: *mut u64,
    ) -> cudf_status_t;

    pub fn cudf_test_io_kvikio_active_io_begin() -> cudf_status_t;

    pub fn cudf_test_io_kvikio_active_io_end() -> cudf_status_t;

    /// Opens a path datasource and reports datasource-level read capability
    /// without launching a Parquet decode.
    pub fn cudf_io_probe_path_datasource(
        filepath: *const c_char,
        preferred_read_size: i64,
        out: *mut cudf_io_datasource_probe_result_t,
    ) -> cudf_status_t;

    /// Opens a scoped remote datasource and reports read capability.
    pub fn cudf_io_probe_remote_parquet_datasource(
        uri: *const c_char,
        profile: *const cudf_io_remote_parquet_profile_t,
        preferred_read_size: i64,
        out: *mut cudf_io_datasource_probe_result_t,
    ) -> cudf_status_t;

    /// Write Parquet with explicit execution context.
    ///
    /// `options` must name an explicit valid device and non-null stream/MR.
    /// The shim installs `options->mr` as the current device resource only for
    /// the duration of the libcudf writer call.
    pub fn cudf_io_write_parquet_ex(
        table: cudf_table_view_handle,
        filepath: *const c_char,
        column_names: *const *const c_char,
        num_columns: i32,
        row_group_size_rows: i32,
        dictionary_policy: i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
