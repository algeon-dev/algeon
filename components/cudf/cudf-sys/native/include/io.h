#ifndef CUDF_RUST_IO_H
#define CUDF_RUST_IO_H

#include "error.h"
#include "execution_context.h"
#include "ast.h"
#include "table.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUDF_TEST_IO_METADATA_FAULT_NONE                = 0,
  CUDF_TEST_IO_METADATA_FAULT_NEGATIVE_NAME_COUNT = 1,
  CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ARRAY     = 2,
  CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ENTRY     = 3,
  CUDF_TEST_IO_METADATA_FAULT_INVALID_UTF8_NAME   = 4,
  CUDF_TEST_IO_METADATA_FAULT_ZERO_NAME_COUNT     = 5,
} cudf_test_io_metadata_fault_t;

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
/// Contract:
/// - `num_columns` must be >= 0
/// - if `num_columns > 0`, `columns` must be non-NULL
/// - `skip_rows` must be >= 0
/// - `num_rows` must be >= -1 (`-1` means read all remaining rows)
/// - if `num_row_groups > 0`, `row_groups` must be non-NULL
/// - if `num_row_groups > 0`, `skip_rows` must be 0 and `num_rows` must be -1
/// Read Parquet with explicit execution context.
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_parquet_ex(const char* filepath,
                                      const char** columns,
                                      int32_t num_columns,
                                      int64_t skip_rows,
                                      int64_t num_rows,
                                      const int32_t* row_groups,
                                      int32_t num_row_groups,
                                      cudf_table_handle* out,
                                      const cudf_exec_options_t* exec_options);

/// Read a Parquet file, returning the table and column names.
/// `out_names` is populated with `out_num_names` null-terminated strings.
/// When `out_num_names` is zero, `*out_names` is NULL.
/// The caller must free the names with `cudf_io_free_column_names()`.
///
/// `row_groups` / `num_row_groups` — same semantics as `cudf_io_read_parquet_ex`.
///
/// Contract:
/// - `num_columns` must be >= 0
/// - if `num_columns > 0`, `columns` must be non-NULL
/// - `skip_rows` must be >= 0
/// - `num_rows` must be >= -1 (`-1` means read all remaining rows)
/// - if `num_row_groups > 0`, `row_groups` must be non-NULL
/// - if `num_row_groups > 0`, `skip_rows` must be 0 and `num_rows` must be -1
/// Read Parquet with metadata and explicit execution context.
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_parquet_with_metadata_ex(const char* filepath,
                                                    const char** columns,
                                                    int32_t num_columns,
                                                    int64_t skip_rows,
                                                    int64_t num_rows,
                                                    const int32_t* row_groups,
                                                    int32_t num_row_groups,
                                                    cudf_table_handle* out_table,
                                                    char*** out_names,
                                                    int32_t* out_num_names,
                                                    const cudf_exec_options_t* exec_options);

/// Read Parquet with metadata, explicit execution context, and an optional
/// libcudf AST reader predicate.
///
/// `predicate` may be NULL. When non-NULL, it must outlive this call.
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_parquet_with_metadata_filter_ex(const char* filepath,
                                                           const char** columns,
                                                           int32_t num_columns,
                                                           int64_t skip_rows,
                                                           int64_t num_rows,
                                                           const int32_t* row_groups,
                                                           int32_t num_row_groups,
                                                           cudf_ast_expression_handle predicate,
                                                           cudf_table_handle* out_table,
                                                           char*** out_names,
                                                           int32_t* out_num_names,
                                                           const cudf_exec_options_t* exec_options);

/// Free column name array returned by metadata readers.
void cudf_io_free_column_names(char** names, int32_t num_names);

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
                                       const char* filepath,
                                       const char** column_names,
                                       int32_t num_columns,
                                       int32_t row_group_size_rows,
                                       const cudf_exec_options_t* exec_options);

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

/// Read a CSV file into a table with explicit execution context.
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_csv_ex(const char* filepath,
                                  char delimiter,
                                  int32_t has_header,
                                  cudf_table_handle* out,
                                  const cudf_exec_options_t* exec_options);

/// Write a table to a CSV file.
/// `column_names` is an array of null-terminated strings (one per column).
///
/// Contract:
/// - `num_columns` must be >= 0
/// - if `num_columns > 0`, `column_names` must be non-NULL
/// Write a table to a CSV file with explicit execution context.
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// libcudf's CSV writer accepts a stream but not an explicit MR; the shim
/// installs `exec_options->mr` as the current device resource only for the
/// duration of the libcudf writer call.
cudf_status_t cudf_io_write_csv_ex(cudf_table_view_handle table,
                                   const char* filepath,
                                   const char** column_names,
                                   int32_t num_columns,
                                   char delimiter,
                                   int32_t include_header,
                                   const cudf_exec_options_t* exec_options);

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

/// Read JSON with explicit execution context.
/// `lines`: 1 for JSON Lines (NDJSON), 0 for standard JSON.
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_json_ex(const char* filepath,
                                   int32_t lines,
                                   cudf_table_handle* out,
                                   const cudf_exec_options_t* exec_options);

/// Read JSON with metadata and explicit execution context.
///
/// `out_names` is populated with `out_num_names` null-terminated strings.
/// The caller must free the names with `cudf_io_free_column_names()`.
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// `exec_options->mr` must outlive the returned table.
cudf_status_t cudf_io_read_json_with_metadata_ex(const char* filepath,
                                                 int32_t lines,
                                                 cudf_table_handle* out_table,
                                                 char*** out_names,
                                                 int32_t* out_num_names,
                                                 const cudf_exec_options_t* exec_options);

/// Write a table to a JSON file.
/// `lines`: 1 for JSON Lines (NDJSON), 0 for standard JSON.
/// `include_nulls`: 1 to output null fields, 0 to omit them.
///
/// Contract:
/// - `num_columns` must be >= 0
/// - if `num_columns > 0`, `column_names` must be non-NULL
/// Write JSON with explicit execution context.
///
/// `exec_options` must name an explicit valid device and non-NULL stream/MR.
/// libcudf's JSON writer accepts a stream but not an explicit MR; the shim
/// installs `exec_options->mr` as the current device resource only for the
/// duration of the libcudf writer call.
cudf_status_t cudf_io_write_json_ex(cudf_table_view_handle table,
                                    const char* filepath,
                                    const char** column_names,
                                    int32_t num_columns,
                                    int32_t lines,
                                    int32_t include_nulls,
                                    const cudf_exec_options_t* exec_options);

/// Arm a one-shot malformed metadata response for the next JSON/Parquet
/// metadata read on the current thread. Intended only for regression tests.
void cudf_test_io_set_next_metadata_fault(cudf_test_io_metadata_fault_t fault);

/// Clear any armed malformed metadata response on the current thread.
void cudf_test_io_reset_metadata_fault();

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_IO_H
