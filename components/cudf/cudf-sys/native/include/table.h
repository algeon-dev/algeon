#ifndef CUDF_RUST_TABLE_H
#define CUDF_RUST_TABLE_H

#include "column.h"
#include "error.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a cudf::table (owning).
typedef void* cudf_table_handle;

/// Opaque handle to a cudf::table_view (non-owning).
typedef void* cudf_table_view_handle;

/// Create a table from an array of column handles. The columns are *moved*
/// into the table — callers must NOT destroy the column handles afterwards.
cudf_status_t cudf_table_create(cudf_column_handle* columns, int32_t num_columns,
                                cudf_table_handle* out);

/// Deep-copy a table.
cudf_status_t cudf_table_copy(cudf_table_handle src, cudf_table_handle* out);

/// Destroy an owning table handle.
void cudf_table_destroy(cudf_table_handle handle);

/// Get an immutable view of a table. The view is heap-allocated and must
/// be freed with cudf_table_view_destroy. Borrows from `handle`.
cudf_status_t cudf_table_view(cudf_table_handle handle,
                              cudf_table_view_handle* out);

/// Destroy a table_view handle.
void cudf_table_view_destroy(cudf_table_view_handle handle);

/// Get the number of columns in a table_view.
cudf_status_t cudf_table_view_num_columns(cudf_table_view_handle handle,
                                          int32_t* out);

/// Get the number of rows in a table_view.
cudf_status_t cudf_table_view_num_rows(cudf_table_view_handle handle,
                                       int32_t* out);

/// Get a column_view from a table_view by index. The returned view is
/// heap-allocated and must be freed with cudf_column_view_destroy.
cudf_status_t cudf_table_view_column(cudf_table_view_handle handle,
                                     int32_t index,
                                     cudf_column_view_handle* out);

/// Construct a table_view from an array of column_view handles.
/// The returned table_view is heap-allocated and must be freed with
/// cudf_table_view_destroy. It borrows from the column_views, which
/// must remain alive for the lifetime of the returned table_view.
cudf_status_t cudf_table_view_from_column_views(
    cudf_column_view_handle* column_views,
    int32_t num_columns,
    cudf_table_view_handle* out);

/// Release all columns from a table, destroying the table in the process.
///
/// The table is consumed: after a successful call, `table` is destroyed and
/// must NOT be used again. The caller receives ownership of each column via
/// `out_columns`, which is a heap-allocated array of `out_count` column
/// handles. Each column must be individually destroyed with
/// `cudf_column_destroy`. The array itself must be freed with
/// `cudf_free_column_array` declared below.
///
/// On failure, `table` is NOT consumed (caller must still destroy it).
cudf_status_t cudf_table_release_columns(cudf_table_handle table,
                                         cudf_column_handle** out_columns,
                                         int32_t* out_count);

/// Free a heap-allocated array of column handles returned by APIs such as
/// cudf_table_release_columns and cudf_groupby_aggregate.
/// Does NOT destroy the individual columns — caller must do that first.
void cudf_free_column_array(cudf_column_handle* array);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_TABLE_H
