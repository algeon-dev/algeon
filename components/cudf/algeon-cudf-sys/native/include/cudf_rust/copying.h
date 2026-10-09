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
  CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY = 0,
  CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK = 1,
} cudf_out_of_bounds_policy_t;

/// Gather rows from a table by index with explicit execution options.
///
/// `gather_map` must be a non-nullable INT32 column of row indices.
///
/// `options` must name an explicit valid device and non-NULL stream/MR.
/// `options->mr` must outlive the returned table.
cudf_status_t cudf_gather_ex(cudf_table_view_handle input,
                             cudf_column_view_handle gather_map,
                             cudf_out_of_bounds_policy_t bounds_policy,
                             cudf_table_handle *out,
                             cudf_exec_options_t const *options);

/// Return the exact device allocation size of gathering `input` by
/// `gather_map`.
///
/// This executes libcudf gather through the selected stream/memory resource,
/// observes the resulting table allocation size, and releases the temporary
/// table before returning. The value is therefore the same allocation size
/// that `cudf_gather_ex` would retain for the materialized output.
/// `options` must name an explicit valid device and non-NULL stream/MR.
cudf_status_t cudf_gather_device_allocated_size_ex(
    cudf_table_view_handle input, cudf_column_view_handle gather_map,
    cudf_out_of_bounds_policy_t bounds_policy, size_t *out,
    cudf_exec_options_t const *options);

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
                              cudf_table_handle *out,
                              cudf_exec_options_t const *options);

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
                                      cudf_column_handle *out,
                                      cudf_exec_options_t const *options);

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
cudf_status_t cudf_shift_ex(cudf_column_view_handle input, int32_t offset,
                            cudf_scalar_handle fill_value,
                            cudf_column_handle *out,
                            cudf_exec_options_t const *options);

#ifdef __cplusplus
}
#endif

#endif // CUDF_RUST_COPYING_H
