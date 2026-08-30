#ifndef CUDF_RUST_REDUCTION_H
#define CUDF_RUST_REDUCTION_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "scalar.h"
#include "types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a cudf::reduce_aggregation.
typedef struct cudf_reduce_aggregation_handle_state* cudf_reduce_aggregation_handle;

/// Opaque handle to a cudf::scan_aggregation.
typedef struct cudf_scan_aggregation_handle_state* cudf_scan_aggregation_handle;

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

/// Allocation-free workspace bytes for a fixed-width, single-column distinct count.
typedef struct cudf_distinct_count_workspace_preflight_result_t {
  uint64_t preprocessing_retained_bytes;
  uint64_t static_set_storage_bytes;
  uint64_t insertion_count_bytes;
  uint64_t active_workspace_peak_bytes;
} cudf_distinct_count_workspace_preflight_result_t;

/// Compute the native workspace bound for a global distinct count without allocating device memory.
/// Only one non-null, non-nested, fixed-width key column is supported.
cudf_status_t cudf_distinct_count_workspace_preflight(
  int64_t num_rows,
  cudf_type_id_t key_type,
  int64_t null_count,
  int32_t key_count,
  cudf_distinct_count_workspace_preflight_result_t* out);

/// Create a simple reduce aggregation.
cudf_status_t cudf_reduce_aggregation_create(cudf_reduce_aggregation_kind_t kind,
                                             cudf_reduce_aggregation_handle* out);

/// Create a VARIANCE reduce aggregation with ddof.
cudf_status_t cudf_reduce_aggregation_create_variance(int32_t ddof,
                                                      cudf_reduce_aggregation_handle* out);

/// Create a STD reduce aggregation with ddof.
cudf_status_t cudf_reduce_aggregation_create_std(int32_t ddof, cudf_reduce_aggregation_handle* out);

/// Create a NUNIQUE reduce aggregation with explicit null handling.
cudf_status_t cudf_reduce_aggregation_create_nunique(int32_t null_policy,
                                                     cudf_reduce_aggregation_handle* out);

/// Destroy a reduce aggregation handle. Ownership is consumed on call;
/// success means cleanup completed and failure quarantines the handle.
cudf_status_t cudf_reduce_aggregation_destroy(cudf_reduce_aggregation_handle handle);

/// Create a simple scan aggregation.
cudf_status_t cudf_scan_aggregation_create(cudf_reduce_aggregation_kind_t kind,
                                           cudf_scan_aggregation_handle* out);

/// Destroy a scan aggregation handle. Ownership is consumed on call; success
/// means cleanup completed and failure quarantines the handle.
cudf_status_t cudf_scan_aggregation_destroy(cudf_scan_aggregation_handle handle);

/// Reduce a column to a single scalar value with explicit execution context.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned scalar.
cudf_status_t cudf_reduce_ex(cudf_column_view_handle input,
                             cudf_reduce_aggregation_handle agg,
                             cudf_data_type_handle output_type,
                             cudf_scalar_handle* out,
                             cudf_exec_options_t const* options);

/// Compute a prefix scan on a column with explicit execution context.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_scan_ex(cudf_column_view_handle input,
                           cudf_scan_aggregation_handle agg,
                           cudf_scan_type_t scan_type,
                           int32_t null_policy,
                           cudf_column_handle* out,
                           cudf_exec_options_t const* options);

/// Compute the min and max of a column with explicit execution context.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned scalars.
cudf_status_t cudf_minmax_ex(cudf_column_view_handle input,
                             cudf_scalar_handle* out_min,
                             cudf_scalar_handle* out_max,
                             cudf_exec_options_t const* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_REDUCTION_H
