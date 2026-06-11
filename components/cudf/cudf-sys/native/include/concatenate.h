#ifndef CUDF_RUST_CONCATENATE_H
#define CUDF_RUST_CONCATENATE_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "table.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Concatenate multiple columns into a single column.
cudf_status_t cudf_concatenate_columns(const cudf_column_view_handle* columns,
                                       int32_t num_columns,
                                       cudf_column_handle* out);

/// Concatenate multiple columns into a single column with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_concatenate_columns_ex(const cudf_column_view_handle* columns,
                                          int32_t num_columns,
                                          const cudf_exec_options_t* options,
                                          cudf_column_handle* out);

/// Concatenate multiple tables into a single table.
cudf_status_t cudf_concatenate_tables(const cudf_table_view_handle* tables,
                                      int32_t num_tables,
                                      cudf_table_handle* out);

/// Concatenate multiple tables into a single table with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned table.
cudf_status_t cudf_concatenate_tables_ex(const cudf_table_view_handle* tables,
                                         int32_t num_tables,
                                         const cudf_exec_options_t* options,
                                         cudf_table_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_CONCATENATE_H
