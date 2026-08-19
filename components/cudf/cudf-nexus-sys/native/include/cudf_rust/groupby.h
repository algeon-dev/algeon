#ifndef CUDF_RUST_GROUPBY_H
#define CUDF_RUST_GROUPBY_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "reduction.h"
#include "scalar.h"
#include "table.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a cudf::groupby_aggregation.
typedef struct cudf_aggregation_handle_state* cudf_aggregation_handle;

/// Opaque handle to a cudf::groupby_scan_aggregation.
typedef struct cudf_groupby_scan_aggregation_handle_state* cudf_groupby_scan_aggregation_handle;

/// Opaque handle to a reusable cudf::groupby::groupby object.
typedef struct cudf_prepared_groupby_handle_state* cudf_prepared_groupby_handle;

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
cudf_status_t cudf_aggregation_create(cudf_aggregation_kind_t kind, cudf_aggregation_handle* out);

/// Create a COUNT aggregation with explicit null handling.
cudf_status_t cudf_aggregation_create_count(cudf_null_policy_t null_policy,
                                            cudf_aggregation_handle* out);

/// Create a STD aggregation with ddof.
cudf_status_t cudf_aggregation_create_std(int32_t ddof, cudf_aggregation_handle* out);

/// Create a VARIANCE aggregation with ddof.
cudf_status_t cudf_aggregation_create_variance(int32_t ddof, cudf_aggregation_handle* out);

/// Destroy an aggregation handle. Ownership is consumed on call; success
/// means cleanup completed and failure means the handle remains quarantined.
cudf_status_t cudf_aggregation_destroy(cudf_aggregation_handle handle);

/// Create a grouped scan aggregation (SUM, MIN, MAX, PRODUCT).
cudf_status_t cudf_groupby_scan_aggregation_create(cudf_reduce_aggregation_kind_t kind,
                                                   cudf_groupby_scan_aggregation_handle* out);

/// Destroy a grouped scan aggregation handle. Ownership is consumed on call;
/// success means cleanup completed and failure means the handle is quarantined.
cudf_status_t cudf_groupby_scan_aggregation_destroy(cudf_groupby_scan_aggregation_handle handle);

/// Create a reusable groupby handle for a key table.
///
/// The returned handle borrows the columns referenced by `keys`; those columns
/// must outlive the handle. If `keys_are_sorted` is non-zero, the caller asserts
/// that key columns are ascending with nulls after, matching cuDF's pre-sorted
/// key contract.
cudf_status_t cudf_prepared_groupby_create_ex(cudf_table_view_handle keys,
                                              cudf_null_policy_t keys_null_policy,
                                              int32_t keys_are_sorted,
                                              cudf_prepared_groupby_handle* out,
                                              cudf_exec_options_t const* options);

/// Destroy a reusable groupby handle. Ownership is consumed on call; success
/// means cleanup completed and failure means the handle remains quarantined.
cudf_status_t cudf_prepared_groupby_destroy(cudf_prepared_groupby_handle handle);

/// Prepare reusable grouping state and return stable cardinality facts.
///
/// This calls cuDF's `groupby::get_groups` without value columns. The returned
/// `out_num_groups` is derived from cuDF's group offsets after the handle's
/// null-key policy is applied. `out_grouped_key_output_bytes` reports the exact
/// device allocation footprint of the grouped-key table materialized by this
/// call, and `out_max_group_size` is computed from cuDF's host group offsets.
/// Temporary grouped-key output is released before this call returns; these are
/// grouped-output facts, not retained prepared-handle state bytes. The prepared
/// groupby handle remains reusable for subsequent aggregate calls.
cudf_status_t cudf_prepared_groupby_prepare_groups_ex(cudf_prepared_groupby_handle groupby,
                                                      int64_t* out_num_groups,
                                                      size_t* out_grouped_key_output_bytes,
                                                      int64_t* out_max_group_size,
                                                      cudf_exec_options_t const* options);

/// Execute an aggregation through a reusable groupby handle.
///
/// The handle borrows its key columns; value columns and aggregation handles
/// are borrowed for the duration of this call only. `options` must name an
/// explicit valid device and non-NULL stream/MR. `options->mr` must outlive the
/// returned keys table and every returned result column.
cudf_status_t cudf_prepared_groupby_aggregate_ex(cudf_prepared_groupby_handle groupby,
                                                 cudf_column_view_handle const* values,
                                                 cudf_aggregation_handle const* aggregations,
                                                 int32_t const* agg_counts,
                                                 int32_t num_requests,
                                                 int32_t num_aggregations,
                                                 cudf_table_handle* out_keys,
                                                 cudf_column_handle** out_values,
                                                 int32_t* out_total_columns,
                                                 cudf_exec_options_t const* options);

/// Perform a groupby aggregation with explicit execution context.
///
/// @param keys           Table view of grouping key columns.
/// @param values         Array of column view handles for value columns.
/// @param aggregations   Flat array of aggregation handles. For request `i`,
///                        the aggregations start at offset sum(agg_counts[0..i]).
/// @param agg_counts     Array of length `num_requests`. `agg_counts[i]` is the
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
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned keys table and every returned
/// result column.
cudf_status_t cudf_groupby_aggregate_ex(cudf_table_view_handle keys,
                                        cudf_column_view_handle const* values,
                                        cudf_aggregation_handle const* aggregations,
                                        int32_t const* agg_counts,
                                        int32_t num_requests,
                                        int32_t num_aggregations,
                                        cudf_null_policy_t keys_null_policy,
                                        cudf_table_handle* out_keys,
                                        cudf_column_handle** out_values,
                                        int32_t* out_total_columns,
                                        cudf_exec_options_t const* options);

/// Perform a groupby aggregation with explicit execution context and sorted-key
/// contract.
///
/// If `keys_are_sorted` is non-zero, the caller asserts that key columns are
/// ascending with nulls after, matching cuDF's pre-sorted key contract.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned keys table and every returned
/// result column.
cudf_status_t cudf_groupby_aggregate_with_keys_are_sorted_ex(
  cudf_table_view_handle keys,
  cudf_column_view_handle const* values,
  cudf_aggregation_handle const* aggregations,
  int32_t const* agg_counts,
  int32_t num_requests,
  int32_t num_aggregations,
  cudf_null_policy_t keys_null_policy,
  int32_t keys_are_sorted,
  cudf_table_handle* out_keys,
  cudf_column_handle** out_values,
  int32_t* out_total_columns,
  cudf_exec_options_t const* options);

/// Perform a grouped prefix scan with explicit execution context.
///
/// The output has one result column per requested aggregation. If
/// `keys_are_sorted` is non-zero, the grouped scan is constructed with cuDF's
/// sorted-key contract and assumes ascending key columns with nulls after.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned keys table and every returned
/// result column.
cudf_status_t cudf_groupby_scan_ex(cudf_table_view_handle keys,
                                   cudf_column_view_handle const* values,
                                   cudf_groupby_scan_aggregation_handle const* aggregations,
                                   int32_t const* agg_counts,
                                   int32_t num_requests,
                                   int32_t num_aggregations,
                                   cudf_null_policy_t keys_null_policy,
                                   int32_t keys_are_sorted,
                                   cudf_table_handle* out_keys,
                                   cudf_column_handle** out_values,
                                   int32_t* out_total_columns,
                                   cudf_exec_options_t const* options);

/// Perform a per-group row shift with explicit execution context.
///
/// Each value column is shifted by the corresponding `offsets[i]` positions
/// within each group, and indeterminate output positions take `fill_values[i]`.
/// libcudf returns a keys table and a values table with identical row counts.
/// Within each group the rows of the values table appear in the input row
/// order; the order of groups themselves is libcudf-defined and may differ
/// across calls.
///
/// `values`, `offsets`, and `fill_values` must all have length
/// `num_value_columns`. Each `fill_values[i]` must have the same data type as
/// the corresponding `values[i]` column. All value columns must have the same
/// row count. libcudf rejects value columns whose data type is neither
/// fixed-width nor string.
///
/// If `keys_are_sorted` is non-zero, the caller asserts that the key columns
/// are ascending with nulls after, matching cuDF's pre-sorted key contract.
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned keys and values tables.
cudf_status_t cudf_groupby_shift_ex(cudf_table_view_handle keys,
                                    cudf_column_view_handle const* values,
                                    int32_t const* offsets,
                                    cudf_scalar_handle const* fill_values,
                                    int32_t num_value_columns,
                                    cudf_null_policy_t keys_null_policy,
                                    int32_t keys_are_sorted,
                                    cudf_table_handle* out_keys,
                                    cudf_table_handle* out_values,
                                    cudf_exec_options_t const* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_GROUPBY_H
