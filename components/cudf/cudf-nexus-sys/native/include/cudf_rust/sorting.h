#ifndef CUDF_RUST_SORTING_H
#define CUDF_RUST_SORTING_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "table.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Column sort order.
typedef enum {
  CUDF_ORDER_ASCENDING  = 0,
  CUDF_ORDER_DESCENDING = 1,
} cudf_order_t;

/// Null ordering within sorted output.
typedef enum {
  CUDF_NULL_ORDER_AFTER  = 0,
  CUDF_NULL_ORDER_BEFORE = 1,
} cudf_null_order_t;

/// Allocation components for the unstable one-key radix sorted-order path.
///
/// All fields are requested bytes. `key_input_bytes` is zero for integral and
/// chrono keys and the transformed float-pair input for floating keys.
/// `active_phase_peak_bytes` is the checked sum of the five components, which
/// are co-resident during execution.
typedef struct cudf_sorted_order_radix_preflight_result_t {
  uint64_t key_input_bytes;
  uint64_t key_output_bytes;
  uint64_t sequence_bytes;
  uint64_t temporary_workspace_bytes;
  uint64_t retained_order_bytes;
  uint64_t active_phase_peak_bytes;
} cudf_sorted_order_radix_preflight_result_t;

/// Metadata-only preflight for the unstable one-key radix sorted-order path.
///
/// `key_type` uses the `cudf_type_id_t` discriminants. The selected candidate
/// device is activated only for the CUB temporary-storage query and restored
/// before return. No stream, memory resource, column, or device allocation is
/// required.
cudf_status_t cudf_sorted_order_radix_preflight(int64_t num_rows,
                                                cudf_type_id_t key_type,
                                                int64_t null_count,
                                                int32_t key_count,
                                                int32_t stable,
                                                cudf_order_t key_order,
                                                int32_t device,
                                                cudf_sorted_order_radix_preflight_result_t* out);

/// Sort with explicit execution context.
///
/// `column_order` and `null_precedence` are arrays of size `num_keys`
/// (i.e., the number of columns in `input`). Pass NULL for defaults
/// (all ASCENDING, nulls AFTER).
/// `options` must be non-NULL and carry an explicit device, stream, and memory
/// resource. `options->mr` must outlive the returned table.
cudf_status_t cudf_sort_ex(cudf_table_view_handle input,
                           const cudf_order_t* column_order,
                           const cudf_null_order_t* null_precedence,
                           int32_t num_keys,
                           cudf_table_handle* out,
                           const cudf_exec_options_t* options);

/// Stable sort with explicit execution context.
/// `options` must be non-NULL and carry an explicit device, stream, and memory
/// resource. `options->mr` must outlive the returned table.
cudf_status_t cudf_stable_sort_ex(cudf_table_view_handle input,
                                  const cudf_order_t* column_order,
                                  const cudf_null_order_t* null_precedence,
                                  int32_t num_keys,
                                  cudf_table_handle* out,
                                  const cudf_exec_options_t* options);

/// Sort by key with explicit execution context.
/// `options` must be non-NULL and carry an explicit device, stream, and memory
/// resource. `options->mr` must outlive the returned table.
cudf_status_t cudf_sort_by_key_ex(cudf_table_view_handle values,
                                  cudf_table_view_handle keys,
                                  const cudf_order_t* column_order,
                                  const cudf_null_order_t* null_precedence,
                                  int32_t num_key_columns,
                                  cudf_table_handle* out,
                                  const cudf_exec_options_t* options);

/// Stable sort by key with explicit execution context.
/// `options` must be non-NULL and carry an explicit device, stream, and memory
/// resource. `options->mr` must outlive the returned table.
cudf_status_t cudf_stable_sort_by_key_ex(cudf_table_view_handle values,
                                         cudf_table_view_handle keys,
                                         const cudf_order_t* column_order,
                                         const cudf_null_order_t* null_precedence,
                                         int32_t num_key_columns,
                                         cudf_table_handle* out,
                                         const cudf_exec_options_t* options);

/// Sorted order with explicit execution context.
///
/// Output is a single INT32 column.
/// `options` must be non-NULL and carry an explicit device, stream, and memory
/// resource. `options->mr` must outlive the returned column.
cudf_status_t cudf_sorted_order_ex(cudf_table_view_handle input,
                                   const cudf_order_t* column_order,
                                   const cudf_null_order_t* null_precedence,
                                   int32_t num_keys,
                                   cudf_column_handle* out,
                                   const cudf_exec_options_t* options);

/// Check whether the table is already sorted with explicit execution context.
///
/// `options` must be non-NULL and carry an explicit device, stream, and memory
/// resource.
cudf_status_t cudf_is_sorted_ex(cudf_table_view_handle input,
                                const cudf_order_t* column_order,
                                const cudf_null_order_t* null_precedence,
                                int32_t num_keys,
                                int32_t* out,
                                const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_SORTING_H
