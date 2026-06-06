#ifndef CUDF_RUST_REDUCTION_H
#define CUDF_RUST_REDUCTION_H

#include "error.h"
#include "column.h"
#include "execution_context.h"
#include "scalar.h"
#include "types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a cudf::reduce_aggregation.
typedef void* cudf_reduce_aggregation_handle;

/// Opaque handle to a cudf::scan_aggregation.
typedef void* cudf_scan_aggregation_handle;

/// Reduction aggregation kinds (shared by reduce and scan).
typedef enum {
  CUDF_REDUCE_AGG_SUM = 0,
  CUDF_REDUCE_AGG_MIN,
  CUDF_REDUCE_AGG_MAX,
  CUDF_REDUCE_AGG_PRODUCT,
  CUDF_REDUCE_AGG_MEAN,
  CUDF_REDUCE_AGG_MEDIAN,
  CUDF_REDUCE_AGG_ANY,
  CUDF_REDUCE_AGG_ALL,
} cudf_reduce_aggregation_kind_t;

/// Scan type: inclusive or exclusive.
typedef enum {
  CUDF_SCAN_INCLUSIVE = 0,
  CUDF_SCAN_EXCLUSIVE = 1,
} cudf_scan_type_t;

/// Create a simple reduce aggregation.
cudf_status_t cudf_reduce_aggregation_create(
    cudf_reduce_aggregation_kind_t kind,
    cudf_reduce_aggregation_handle* out);

/// Create a VARIANCE reduce aggregation with ddof.
cudf_status_t cudf_reduce_aggregation_create_variance(
    int32_t ddof,
    cudf_reduce_aggregation_handle* out);

/// Create a STD reduce aggregation with ddof.
cudf_status_t cudf_reduce_aggregation_create_std(
    int32_t ddof,
    cudf_reduce_aggregation_handle* out);

/// Destroy a reduce aggregation handle.
void cudf_reduce_aggregation_destroy(cudf_reduce_aggregation_handle handle);

/// Create a simple scan aggregation.
cudf_status_t cudf_scan_aggregation_create(
    cudf_reduce_aggregation_kind_t kind,
    cudf_scan_aggregation_handle* out);

/// Destroy a scan aggregation handle.
void cudf_scan_aggregation_destroy(cudf_scan_aggregation_handle handle);

/// Reduce a column to a single scalar value.
cudf_status_t cudf_reduce(cudf_column_view_handle input,
                          cudf_reduce_aggregation_handle agg,
                          cudf_data_type_handle output_type,
                          cudf_scalar_handle* out);

/// Reduce a column to a single scalar value with explicit execution context.
/// When `options->mr` is non-NULL, it must outlive the returned scalar.
cudf_status_t cudf_reduce_ex(cudf_column_view_handle input,
                             cudf_reduce_aggregation_handle agg,
                             cudf_data_type_handle output_type,
                             cudf_scalar_handle* out,
                             const cudf_exec_options_t* options);

/// Compute a prefix scan (cumulative operation) on a column.
cudf_status_t cudf_scan(cudf_column_view_handle input,
                        cudf_scan_aggregation_handle agg,
                        cudf_scan_type_t scan_type,
                        int32_t null_policy,
                        cudf_column_handle* out);

/// Compute a prefix scan on a column with explicit execution context.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_scan_ex(cudf_column_view_handle input,
                           cudf_scan_aggregation_handle agg,
                           cudf_scan_type_t scan_type,
                           int32_t null_policy,
                           cudf_column_handle* out,
                           const cudf_exec_options_t* options);

/// Compute the min and max of a column simultaneously.
cudf_status_t cudf_minmax(cudf_column_view_handle input,
                          cudf_scalar_handle* out_min,
                          cudf_scalar_handle* out_max);

/// Compute the min and max of a column with explicit execution context.
/// When `options->mr` is non-NULL, it must outlive the returned scalars.
cudf_status_t cudf_minmax_ex(cudf_column_view_handle input,
                             cudf_scalar_handle* out_min,
                             cudf_scalar_handle* out_max,
                             const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_REDUCTION_H
