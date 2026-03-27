#ifndef CUDF_RUST_SORTING_H
#define CUDF_RUST_SORTING_H

#include "error.h"
#include "table.h"
#include "column.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Column sort order.
typedef enum {
  CUDF_ORDER_ASCENDING = 0,
  CUDF_ORDER_DESCENDING = 1,
} cudf_order_t;

/// Null ordering within sorted output.
typedef enum {
  CUDF_NULL_ORDER_AFTER = 0,
  CUDF_NULL_ORDER_BEFORE = 1,
} cudf_null_order_t;

/// Sort the rows of a table by all columns.
/// `column_order` and `null_precedence` are arrays of size `num_keys`
/// (i.e., the number of columns in `input`). Pass NULL for defaults
/// (all ASCENDING, nulls AFTER).
cudf_status_t cudf_sort(cudf_table_view_handle input,
                        const cudf_order_t* column_order,
                        const cudf_null_order_t* null_precedence,
                        int32_t num_keys,
                        cudf_table_handle* out);

/// Stable sort (preserves relative order of equal elements).
cudf_status_t cudf_stable_sort(cudf_table_view_handle input,
                               const cudf_order_t* column_order,
                               const cudf_null_order_t* null_precedence,
                               int32_t num_keys,
                               cudf_table_handle* out);

/// Sort `values` table by a separate `keys` table (unstable).
cudf_status_t cudf_sort_by_key(cudf_table_view_handle values,
                               cudf_table_view_handle keys,
                               const cudf_order_t* column_order,
                               const cudf_null_order_t* null_precedence,
                               int32_t num_key_columns,
                               cudf_table_handle* out);

/// Stable sort `values` table by a separate `keys` table.
cudf_status_t cudf_stable_sort_by_key(cudf_table_view_handle values,
                                      cudf_table_view_handle keys,
                                      const cudf_order_t* column_order,
                                      const cudf_null_order_t* null_precedence,
                                      int32_t num_key_columns,
                                      cudf_table_handle* out);

/// Return the row indices that would sort the table.
/// Output is a single INT32 column.
cudf_status_t cudf_sorted_order(cudf_table_view_handle input,
                                const cudf_order_t* column_order,
                                const cudf_null_order_t* null_precedence,
                                int32_t num_keys,
                                cudf_column_handle* out);

/// Check whether the table is already sorted.
cudf_status_t cudf_is_sorted(cudf_table_view_handle input,
                             const cudf_order_t* column_order,
                             const cudf_null_order_t* null_precedence,
                             int32_t num_keys,
                             int32_t* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_SORTING_H
