#ifndef CUDF_RUST_CONCATENATE_H
#define CUDF_RUST_CONCATENATE_H

#include "error.h"
#include "column.h"
#include "table.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Concatenate multiple columns into a single column.
cudf_status_t cudf_concatenate_columns(
    const cudf_column_view_handle* columns,
    int32_t num_columns,
    cudf_column_handle* out);

/// Concatenate multiple tables into a single table.
cudf_status_t cudf_concatenate_tables(
    const cudf_table_view_handle* tables,
    int32_t num_tables,
    cudf_table_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_CONCATENATE_H
