#ifndef CUDF_RUST_IO_H
#define CUDF_RUST_IO_H

#include "ast.h"
#include "error.h"
#include "execution_context.h"
#include "table.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t can_open;
  int32_t supports_device_read;
  int32_t device_read_preferred;
  int64_t size_bytes;
} cudf_io_datasource_probe_result_t;

typedef struct {
  char const* aws_region;
  char const* aws_access_key_id;
  char const* aws_secret_access_key;
  char const* aws_session_token;
  char const* aws_endpoint_url;
  /// -1 = unset, 0 = virtual-hosted style, 1 = path-style.
  int32_t path_style_access;
  /// 1 = unsigned/public request, 0 = signed/default-chain request.
  int32_t unsigned_request;
  /// Full object size in bytes when known by the caller, or -1 when unknown.
  int64_t known_size_bytes;
} cudf_io_remote_parquet_profile_t;

typedef struct {
  int32_t backend;
  int32_t fallback_reason;
  int32_t supports_device_read;
  int32_t device_read_used;
  int64_t remote_bytes;
  int64_t remote_range_count;
  int64_t host_read_bytes;
  int64_t host_read_count;
  int64_t device_read_bytes;
  int64_t device_read_count;
} cudf_io_remote_parquet_read_evidence_t;

typedef struct {
  int64_t input_row_groups;
  int32_t has_row_groups_after_stats_filter;
  int64_t row_groups_after_stats_filter;
  int32_t has_row_groups_after_bloom_filter;
  int64_t row_groups_after_bloom_filter;
} cudf_io_parquet_pruning_evidence_t;

typedef enum {
  CUDF_IO_KVIKIO_COMPAT_MODE_AUTO = 0,
  CUDF_IO_KVIKIO_COMPAT_MODE_ON   = 1,
  CUDF_IO_KVIKIO_COMPAT_MODE_OFF  = 2,
} cudf_io_kvikio_compat_mode_t;

// ---------------------------------------------------------------------------
// Parquet
// ---------------------------------------------------------------------------

/// Read a Parquet file into a table. `columns` is an optional array of
/// null-terminated column name strings (pass NULL to read all columns).
/// `num_columns` is the length of the `columns` array.
///
/// `row_groups` is an optional array of zero-based row group indices to read
/// (for the single source file). Pass NULL with `num_row_groups` = 0 to read
/// all row groups. Cannot be combined with `skip_rows > 0` or `num_rows >= 0`.
///
/// Read Parquet with explicit execution context.
///
/// Contract:
/// - `filepath` must be non-NULL
/// - `num_columns` must be >= 0
/// - if `num_columns > 0`, `columns` must be non-NULL
/// - `skip_rows` must be >= 0
/// - `num_rows` must be >= -1 (`-1` means read all remaining rows)
/// - if `num_row_groups > 0`, `row_groups` must be non-NULL
/// - if `num_row_groups > 0`, `skip_rows` must be 0 and `num_rows` must be -1
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_parquet_ex(char const* filepath,
                                      char const** columns,
                                      int32_t num_columns,
                                      int64_t skip_rows,
                                      int64_t num_rows,
                                      int32_t const* row_groups,
                                      int32_t num_row_groups,
                                      cudf_table_handle* out,
                                      cudf_exec_options_t const* exec_options);

/// Read a Parquet payload borrowed from host memory into a table.
///
/// `host_buffer` is borrowed only for the duration of this call. The returned
/// table must not retain a pointer into the host buffer.
///
/// `host_buffer` must be non-NULL and `host_buffer_len` must be > 0.
/// `columns`, `skip_rows`, `num_rows`, `row_groups`, and `exec_options` have
/// the same contract as `cudf_io_read_parquet_ex`.
cudf_status_t cudf_io_read_parquet_from_host_buffer_ex(void const* host_buffer,
                                                       size_t host_buffer_len,
                                                       char const** columns,
                                                       int32_t num_columns,
                                                       int64_t skip_rows,
                                                       int64_t num_rows,
                                                       int32_t const* row_groups,
                                                       int32_t num_row_groups,
                                                       cudf_table_handle* out,
                                                       cudf_exec_options_t const* exec_options);

/// Read a Parquet file, returning the table and column names.
/// `out_names` is populated with `out_num_names` null-terminated strings.
/// When `out_num_names` is zero, `*out_names` is NULL.
/// The caller must free the names with `cudf_io_free_column_names()`.
///
/// `row_groups` / `num_row_groups`; same semantics as `cudf_io_read_parquet_ex`.
///
/// Read Parquet with metadata and explicit execution context.
///
/// Contract:
/// - `filepath` must be non-NULL
/// - `num_columns` must be >= 0
/// - if `num_columns > 0`, `columns` must be non-NULL
/// - `skip_rows` must be >= 0
/// - `num_rows` must be >= -1 (`-1` means read all remaining rows)
/// - if `num_row_groups > 0`, `row_groups` must be non-NULL
/// - if `num_row_groups > 0`, `skip_rows` must be 0 and `num_rows` must be -1
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_parquet_with_metadata_ex(char const* filepath,
                                                    char const** columns,
                                                    int32_t num_columns,
                                                    int64_t skip_rows,
                                                    int64_t num_rows,
                                                    int32_t const* row_groups,
                                                    int32_t num_row_groups,
                                                    cudf_table_handle* out_table,
                                                    char*** out_names,
                                                    int32_t* out_num_names,
                                                    cudf_exec_options_t const* exec_options);

/// Read Parquet with metadata, explicit execution context, and an optional
/// libcudf AST reader predicate.
///
/// `predicate` may be NULL. When non-NULL, it must outlive this call.
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_parquet_with_metadata_filter_ex(
  char const* filepath,
  char const** columns,
  int32_t num_columns,
  int64_t skip_rows,
  int64_t num_rows,
  int32_t const* row_groups,
  int32_t num_row_groups,
  cudf_ast_expression_handle predicate,
  cudf_table_handle* out_table,
  char*** out_names,
  int32_t* out_num_names,
  cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
  cudf_exec_options_t const* exec_options);

/// Read remote Parquet through an explicit KvikIO-backed datasource with
/// scoped, non-global S3 options and read evidence.
cudf_status_t cudf_io_read_remote_parquet_with_metadata_filter_ex(
  char const* uri,
  cudf_io_remote_parquet_profile_t const* profile,
  char const** columns,
  int32_t num_columns,
  int64_t skip_rows,
  int64_t num_rows,
  int32_t const* row_groups,
  int32_t num_row_groups,
  cudf_ast_expression_handle predicate,
  cudf_table_handle* out_table,
  char*** out_names,
  int32_t* out_num_names,
  cudf_io_remote_parquet_read_evidence_t* out_evidence,
  cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
  cudf_exec_options_t const* exec_options);

/// Read multiple Parquet files with metadata, explicit execution context, and
/// an optional libcudf AST reader predicate.
///
/// `filepaths` must contain `num_files` non-NULL path strings. Row order follows
/// the file order and then each file's selected row-group order.
///
/// Per-file row groups are represented by `row_group_offsets` and `row_groups`:
/// `row_group_offsets` has `num_files + 1` entries and
/// `row_groups[row_group_offsets[i]..row_group_offsets[i + 1]]` is the row-group
/// selection for file `i`. Pass NULL offsets with zero row groups to read all
/// row groups from every file.
///
/// `predicate` may be NULL. When non-NULL, it must outlive this call.
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_parquet_multi_with_metadata_filter_ex(
  char const** filepaths,
  int32_t num_files,
  char const** columns,
  int32_t num_columns,
  int32_t const* row_group_offsets,
  int32_t num_row_group_offsets,
  int32_t const* row_groups,
  int32_t num_row_groups,
  cudf_ast_expression_handle predicate,
  cudf_table_handle* out_table,
  char*** out_names,
  int32_t* out_num_names,
  cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
  cudf_exec_options_t const* exec_options);

/// Read multiple Parquet files, allowing selected files to use explicit
/// KvikIO-backed remote datasources with scoped S3 options and aggregate read
/// evidence. `remote_source_flags` has `num_files` entries; non-zero entries
/// use the matching `profiles` entry as a remote datasource, and zero entries
/// use libcudf's ordinary path datasource for that file.
cudf_status_t cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex(
  char const** filepaths,
  int32_t const* remote_source_flags,
  cudf_io_remote_parquet_profile_t const* profiles,
  int32_t num_files,
  char const** columns,
  int32_t num_columns,
  int32_t const* row_group_offsets,
  int32_t num_row_group_offsets,
  int32_t const* row_groups,
  int32_t num_row_groups,
  cudf_ast_expression_handle predicate,
  cudf_table_handle* out_table,
  char*** out_names,
  int32_t* out_num_names,
  cudf_io_remote_parquet_read_evidence_t* out_evidence,
  cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
  cudf_exec_options_t const* exec_options);

/// Read Parquet from a host buffer with metadata, explicit execution context,
/// and an optional libcudf AST reader predicate.
///
/// `host_buffer` is borrowed only for the duration of this call. The returned
/// table must not retain a pointer into the host buffer. `host_buffer` must be
/// non-NULL and `host_buffer_len` must be > 0. Other arguments have the same
/// contract as `cudf_io_read_parquet_with_metadata_filter_ex`.
cudf_status_t cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex(
  void const* host_buffer,
  size_t host_buffer_len,
  char const** columns,
  int32_t num_columns,
  int64_t skip_rows,
  int64_t num_rows,
  int32_t const* row_groups,
  int32_t num_row_groups,
  cudf_ast_expression_handle predicate,
  cudf_table_handle* out_table,
  char*** out_names,
  int32_t* out_num_names,
  cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
  cudf_exec_options_t const* exec_options);

/// Free column name array returned by metadata readers.
void cudf_io_free_column_names(char** names, int32_t num_names);

/// Returns the remote Parquet KvikIO build capability:
/// - 1: libcudf was compiled with CUDF_KVIKIO_REMOTE_IO
/// - 0: libcudf was compiled without CUDF_KVIKIO_REMOTE_IO
/// - -1: reserved for unknown/unreported future states
int32_t cudf_io_remote_parquet_kvikio_remote_io_enabled(void);

/// Configure KvikIO's process default compatibility mode without mutating the
/// process environment. Call before starting any I/O workers.
cudf_status_t cudf_io_kvikio_set_compat_mode(int32_t compat_mode);

/// Return KvikIO's configured process default compatibility mode, or -1 when
/// this build has no KvikIO remote-I/O support.
int32_t cudf_io_kvikio_compat_mode(void);

/// Clear every retained KvikIO bounce-buffer pool allocation.
///
/// When I/O is active, no buffers are cleared and `out_active_io` is nonzero.
/// Repeated idle calls are safe and report zero cleared bytes after the first.
cudf_status_t cudf_io_kvikio_drain_bounce_buffer_pools(uint64_t* out_cleared_bytes,
                                                       uint64_t* out_active_io);

/// Opens a path datasource and reports datasource-level read capability without
/// launching a Parquet decode. `preferred_read_size` controls the
/// `is_device_read_preferred` query; pass -1 when no size is known.
cudf_status_t cudf_io_probe_path_datasource(char const* filepath,
                                            int64_t preferred_read_size,
                                            cudf_io_datasource_probe_result_t* out);

/// Opens a scoped remote datasource and reports datasource-level read
/// capability without launching a Parquet decode. This may contact the remote
/// service to resolve file size.
cudf_status_t cudf_io_probe_remote_parquet_datasource(
  char const* uri,
  cudf_io_remote_parquet_profile_t const* profile,
  int64_t preferred_read_size,
  cudf_io_datasource_probe_result_t* out);

/// Write a table to a Parquet file.
/// `column_names` is an array of null-terminated strings (one per column).
/// `row_group_size_rows` sets the maximum number of rows per row group.
/// Pass 0 to use libcudf's default (1,000,000 rows).
///
/// Contract:
/// - `num_columns` must be >= 0
/// - if `num_columns > 0`, `column_names` must be non-NULL
/// Write Parquet with explicit execution context.
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// libcudf's write_parquet surface accepts a stream but not an explicit MR; the
/// shim installs `exec_options->mr` as the current device resource only for the
/// duration of the libcudf writer call.
cudf_status_t cudf_io_write_parquet_ex(cudf_table_view_handle table,
                                       char const* filepath,
                                       char const** column_names,
                                       int32_t num_columns,
                                       int32_t row_group_size_rows,
                                       int32_t dictionary_policy,
                                       cudf_exec_options_t const* exec_options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_IO_H
