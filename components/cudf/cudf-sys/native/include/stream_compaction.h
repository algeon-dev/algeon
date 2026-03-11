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

/// Which duplicate rows to keep.
typedef enum {
  CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY = 0,
  CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST = 1,
  CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST = 2,
  CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE = 3,
} cudf_duplicate_keep_option_t;

/// Whether null elements compare equal.
typedef enum {
  CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL = 0,
  CUDF_STREAM_COMPACTION_NULL_EQUALITY_UNEQUAL = 1,
} cudf_stream_compaction_null_equality_t;

/// Whether NaN elements compare equal.
typedef enum {
  CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL = 0,
  CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL = 1,
} cudf_stream_compaction_nan_equality_t;

/// Create a new table with distinct rows based on key columns.
cudf_status_t cudf_distinct(cudf_table_view_handle input,
                            const int* keys,
                            int num_keys,
                            cudf_duplicate_keep_option_t keep,
                            cudf_stream_compaction_null_equality_t null_equality,
                            cudf_stream_compaction_nan_equality_t nan_equality,
                            cudf_table_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STREAM_COMPACTION_H
