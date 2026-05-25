#ifndef CUDF_RUST_IO_H
#define CUDF_RUST_IO_H

#include "error.h"
#include "table.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Parquet
// ---------------------------------------------------------------------------

/// Read a Parquet file into a table. `columns` is an optional array of
/// null-terminated column name strings (pass NULL to read all columns).
/// `num_columns` is the length of the `columns` array.
cudf_status_t cudf_io_read_parquet(const char* filepath,
                                   const char** columns,
                                   int32_t num_columns,
                                   int64_t skip_rows, int64_t num_rows,
                                   cudf_table_handle* out);

/// Write a table to a Parquet file.
/// `column_names` is an array of null-terminated strings (one per column).
cudf_status_t cudf_io_write_parquet(cudf_table_view_handle table,
                                    const char* filepath,
                                    const char** column_names,
                                    int32_t num_columns);

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

/// Read a CSV file into a table.
cudf_status_t cudf_io_read_csv(const char* filepath, char delimiter,
                               int32_t has_header, cudf_table_handle* out);

/// Write a table to a CSV file.
/// `column_names` is an array of null-terminated strings (one per column).
cudf_status_t cudf_io_write_csv(cudf_table_view_handle table,
                                const char* filepath,
                                const char** column_names,
                                int32_t num_columns, char delimiter,
                                int32_t include_header);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_IO_H
