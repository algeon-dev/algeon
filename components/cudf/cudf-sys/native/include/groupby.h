#ifndef CUDF_RUST_GROUPBY_H
#define CUDF_RUST_GROUPBY_H

#include "error.h"
#include "execution_context.h"
#include "table.h"
#include "column.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a cudf::groupby_aggregation.
typedef void* cudf_aggregation_handle;

/// Aggregation kinds.
typedef enum {
  CUDF_AGG_SUM = 0,
  CUDF_AGG_MIN,
  CUDF_AGG_MAX,
  CUDF_AGG_COUNT_VALID,
  CUDF_AGG_COUNT_ALL,
  CUDF_AGG_MEAN,
  CUDF_AGG_MEDIAN,
  CUDF_AGG_NUNIQUE,
  CUDF_AGG_STD,
  CUDF_AGG_VARIANCE,
} cudf_aggregation_kind_t;

/// Null handling policy for groupby keys.
typedef enum {
  CUDF_NULL_POLICY_EXCLUDE = 0,
  CUDF_NULL_POLICY_INCLUDE = 1,
} cudf_null_policy_t;

/// Create a simple aggregation (SUM, MIN, MAX, MEAN, MEDIAN, NUNIQUE).
cudf_status_t cudf_aggregation_create(cudf_aggregation_kind_t kind,
                                      cudf_aggregation_handle* out);

/// Create a COUNT aggregation with explicit null handling.
cudf_status_t cudf_aggregation_create_count(cudf_null_policy_t null_policy,
                                            cudf_aggregation_handle* out);

/// Create a STD aggregation with ddof.
cudf_status_t cudf_aggregation_create_std(int32_t ddof,
                                          cudf_aggregation_handle* out);

/// Create a VARIANCE aggregation with ddof.
cudf_status_t cudf_aggregation_create_variance(int32_t ddof,
                                               cudf_aggregation_handle* out);

/// Destroy an aggregation handle.
void cudf_aggregation_destroy(cudf_aggregation_handle handle);

/// Perform a groupby aggregation.
///
/// @param keys           Table view of grouping key columns.
/// @param values         Array of column view handles for value columns.
/// @param aggregations   Flat array of aggregation handles. For request `i`,
///                        the aggregations start at offset sum(agg_counts[0..i]).
/// @param agg_counts     Array of length `num_requests`. agg_counts[i] is the
///                        number of aggregations for value column `i`.
/// @param num_requests   Number of aggregation requests (length of values/agg_counts).
/// @param num_aggregations Total length of the flat `aggregations` array. Must
///                        equal `sum(agg_counts)`.
/// @param keys_null_policy  Null handling for keys (EXCLUDE or INCLUDE).
/// @param out_keys       Output: table of unique key groups. Set to NULL on failure.
/// @param out_values     Output: array of column handles for aggregation results.
///                        The array has `sum(agg_counts)` entries. When the
///                        total is zero, `*out_values` is NULL. Caller must
///                        free each column and then the array with cudf_free_column_array.
/// @param out_total_columns  Output: total number of result columns. Set to 0 on failure.
cudf_status_t cudf_groupby_aggregate(
    cudf_table_view_handle keys,
    const cudf_column_view_handle* values,
    const cudf_aggregation_handle* aggregations,
    const int32_t* agg_counts,
    int32_t num_requests,
    int32_t num_aggregations,
    cudf_null_policy_t keys_null_policy,
    cudf_table_handle* out_keys,
    cudf_column_handle** out_values,
    int32_t* out_total_columns);

/// Groupby aggregate with explicit execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned keys table
/// and every returned result column.
cudf_status_t cudf_groupby_aggregate_ex(
    cudf_table_view_handle keys,
    const cudf_column_view_handle* values,
    const cudf_aggregation_handle* aggregations,
    const int32_t* agg_counts,
    int32_t num_requests,
    int32_t num_aggregations,
    cudf_null_policy_t keys_null_policy,
    cudf_table_handle* out_keys,
    cudf_column_handle** out_values,
    int32_t* out_total_columns,
    const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_GROUPBY_H
