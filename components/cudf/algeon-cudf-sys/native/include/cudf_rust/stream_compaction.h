#ifndef CUDF_RUST_STREAM_COMPACTION_H
#define CUDF_RUST_STREAM_COMPACTION_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "table.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Drop rows from a table where key columns contain nulls with explicit
/// execution context.
///
/// `keys` is an array of column indices, `num_keys` is its length.
/// `keep_threshold` is the minimum number of non-null key values required
/// to keep a row.
///
/// `options` must be non-NULL and must carry an explicit device, stream, and
/// memory resource. `options->mr` must outlive the returned table.
cudf_status_t cudf_drop_nulls_ex(cudf_table_view_handle input,
                                 int const* keys,
                                 int num_keys,
                                 int keep_threshold,
                                 cudf_table_handle* out,
                                 cudf_exec_options_t const* options);

/// Which duplicate rows to keep.
typedef enum {
  CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY   = 0,
  CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST = 1,
  CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST  = 2,
  CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE  = 3,
} cudf_duplicate_keep_option_t;

/// Whether null elements compare equal.
typedef enum {
  CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL   = 0,
  CUDF_STREAM_COMPACTION_NULL_EQUALITY_UNEQUAL = 1,
} cudf_stream_compaction_null_equality_t;

/// Whether NaN elements compare equal.
typedef enum {
  CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL = 0,
  CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL   = 1,
} cudf_stream_compaction_nan_equality_t;

/// Create a new table with distinct rows based on key columns with explicit
/// execution context.
///
/// `options` must be non-NULL and must carry an explicit device, stream, and
/// memory resource. `options->mr` must outlive the returned table.
cudf_status_t cudf_distinct_ex(cudf_table_view_handle input,
                               int const* keys,
                               int num_keys,
                               cudf_duplicate_keep_option_t keep,
                               cudf_stream_compaction_null_equality_t null_equality,
                               cudf_stream_compaction_nan_equality_t nan_equality,
                               cudf_table_handle* out,
                               cudf_exec_options_t const* options);

/// Create a new table by removing consecutive duplicate rows with explicit
/// execution context.
///
/// Equal rows must already be adjacent for all duplicates to be removed.
/// `options` must be non-NULL and must carry an explicit device, stream, and
/// memory resource. `options->mr` must outlive the returned table.
cudf_status_t cudf_unique_ex(cudf_table_view_handle input,
                             int const* keys,
                             int num_keys,
                             cudf_duplicate_keep_option_t keep,
                             cudf_stream_compaction_null_equality_t null_equality,
                             cudf_table_handle* out,
                             cudf_exec_options_t const* options);

/// Return row indices after removing consecutive duplicates in a retained order.
/// `row_order` must be a non-null INT32 permutation of the input row indices.
/// Equal key rows must be adjacent in that order. First/last refer to that order.
/// All input handles and execution resources must remain alive through completion.
cudf_status_t cudf_ordered_unique_indices_ex(cudf_table_view_handle input,
                                    cudf_column_view_handle row_order,
                                    int const* keys,
                                    int num_keys,
                                    cudf_duplicate_keep_option_t keep,
                                    cudf_column_handle* out,
                                    cudf_exec_options_t const* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STREAM_COMPACTION_H
