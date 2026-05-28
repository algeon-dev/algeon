#ifndef CUDF_RUST_STREAM_COMPACTION_H
#define CUDF_RUST_STREAM_COMPACTION_H

#include "error.h"
#include "table.h"
#include "column.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Drop rows from a table where key columns contain nulls.
/// `keys` is an array of column indices, `num_keys` is its length.
/// `keep_threshold` is the minimum number of non-null key values required
/// to keep a row.
cudf_status_t cudf_drop_nulls(cudf_table_view_handle input,
                              const int* keys,
                              int num_keys,
                              int keep_threshold,
                              cudf_table_handle* out);

/// duplicate_keep_option values:
///   0 = KEEP_ANY
///   1 = KEEP_FIRST
///   2 = KEEP_LAST
///   3 = KEEP_NONE
///
/// null_equality values:
///   0 = EQUAL
///   1 = UNEQUAL
///
/// nan_equality values:
///   0 = ALL_EQUAL
///   1 = UNEQUAL

/// Create a new table with distinct rows based on key columns.
cudf_status_t cudf_distinct(cudf_table_view_handle input,
                            const int* keys,
                            int num_keys,
                            int keep,
                            int null_equality,
                            int nan_equality,
                            cudf_table_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STREAM_COMPACTION_H
