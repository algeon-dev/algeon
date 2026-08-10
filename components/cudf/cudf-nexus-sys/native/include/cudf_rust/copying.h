#ifndef CUDF_RUST_COPYING_H
#define CUDF_RUST_COPYING_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "scalar.h"
#include "table.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Out-of-bounds policy for gather operations.
typedef enum {
  CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY    = 0,
  CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK = 1,
} cudf_out_of_bounds_policy_t;

/// Flat source-column metadata for the DONT_CHECK gather preflight.
typedef struct cudf_gather_fixed_width_column_metadata_t {
  cudf_type_id_t type_id;
  int32_t nullable;
  int32_t num_children;
} cudf_gather_fixed_width_column_metadata_t;

/// Allocation components for the flat fixed-width DONT_CHECK gather path.
typedef struct cudf_gather_fixed_width_dont_check_preflight_result_t {
  uint64_t gather_map_bytes;
  uint64_t output_data_bytes;
  uint64_t output_null_mask_bytes;
  uint64_t target_mask_pointer_array_bytes;
  uint64_t source_table_device_view_bytes;
  uint64_t valid_count_array_bytes;
  uint64_t native_temporary_workspace_bytes;
  uint64_t active_phase_peak_bytes;
} cudf_gather_fixed_width_dont_check_preflight_result_t;

/// Query flat fixed-width DONT_CHECK gather allocation components from metadata.
///
/// No stream, memory resource, column, or device allocation is required.
/// `columns` may be NULL only when `num_columns == 0`; each `nullable` value
/// must be 0 or 1, and each `num_children` must be non-negative.
cudf_status_t cudf_gather_fixed_width_dont_check_preflight(
  int64_t output_rows,
  const cudf_gather_fixed_width_column_metadata_t* columns,
  int32_t num_columns,
  int32_t device,
  cudf_gather_fixed_width_dont_check_preflight_result_t* out);

/// Gather rows from a table by index with explicit execution options.
///
/// `gather_map` must be a non-nullable INT32 column of row indices.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned table.
cudf_status_t cudf_gather_ex(cudf_table_view_handle input,
                             cudf_column_view_handle gather_map,
                             cudf_out_of_bounds_policy_t bounds_policy,
                             cudf_table_handle* out,
                             const cudf_exec_options_t* options);

/// Return the exact device allocation footprint of gathering `input` by
/// `gather_map`.
///
/// This executes libcudf gather through the selected stream/memory resource,
/// observes the resulting table allocation size, and releases the temporary
/// table before returning. The value is therefore the same allocation
/// footprint that `cudf_gather_ex` would retain for the materialized output.
/// `options` must name an explicit valid device and non-NULL stream/MR.
cudf_status_t cudf_gather_device_allocated_size_ex(cudf_table_view_handle input,
                                                   cudf_column_view_handle gather_map,
                                                   cudf_out_of_bounds_policy_t bounds_policy,
                                                   size_t* out,
                                                   const cudf_exec_options_t* options);

/// Scatter rows from `source` into `target` by index with explicit execution
/// options.
///
/// `scatter_map` must be a non-nullable integral column. libcudf rejects null
/// maps, but duplicate or out-of-bounds indices are undefined behavior and
/// must be validated by safe callers before using this C entry point.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned table.
cudf_status_t cudf_scatter_ex(cudf_table_view_handle source,
                              cudf_column_view_handle scatter_map,
                              cudf_table_view_handle target,
                              cudf_table_handle* out,
                              const cudf_exec_options_t* options);

/// Conditionally select elements from two columns based on a boolean mask with
/// explicit execution options.
///
/// `output[i] = (boolean_mask.valid(i) && boolean_mask[i]) ? lhs[i] : rhs[i]`
///
/// `lhs` and `rhs` must have the same type and length as `boolean_mask`.
/// `boolean_mask` must be of type BOOL8.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_copy_if_else_vv_ex(cudf_column_view_handle lhs,
                                      cudf_column_view_handle rhs,
                                      cudf_column_view_handle boolean_mask,
                                      cudf_column_handle* out,
                                      const cudf_exec_options_t* options);

/// Shift the rows of `input` by `offset` positions with explicit execution
/// options, filling indeterminate outputs with `fill_value`.
///
/// `output[i] = input[i - offset]` when `i - offset` is in `[0, input.size())`,
/// otherwise `output[i] = fill_value`. A positive `offset` shifts forward
/// (rows move toward larger indices); a negative `offset` shifts backward.
///
/// libcudf requires `input` to be a fixed-width or string column. `fill_value`
/// must have the same data type as `input`. A null fill value forces the
/// output to be nullable. If `input` itself is nullable, the output is also
/// nullable regardless of `fill_value`.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned column.
cudf_status_t cudf_shift_ex(cudf_column_view_handle input,
                            int32_t offset,
                            cudf_scalar_handle fill_value,
                            cudf_column_handle* out,
                            const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_COPYING_H
