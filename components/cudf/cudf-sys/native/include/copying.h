#ifndef CUDF_RUST_COPYING_H
#define CUDF_RUST_COPYING_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "table.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Out-of-bounds policy for gather operations.
typedef enum {
  CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY    = 0,
  CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK = 1,
} cudf_out_of_bounds_policy_t;

/// Gather rows from a table by index.
/// `gather_map` must be a non-nullable INT32 column of row indices.
cudf_status_t cudf_gather(cudf_table_view_handle input,
                          cudf_column_view_handle gather_map,
                          cudf_out_of_bounds_policy_t bounds_policy,
                          cudf_table_handle* out);

/// Gather rows from a table by index with explicit execution options.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned table.
cudf_status_t cudf_gather_ex(cudf_table_view_handle input,
                             cudf_column_view_handle gather_map,
                             cudf_out_of_bounds_policy_t bounds_policy,
                             cudf_table_handle* out,
                             const cudf_exec_options_t* options);

/// Conditionally select elements from two columns based on a boolean mask.
///
/// `output[i] = (boolean_mask.valid(i) && boolean_mask[i]) ? lhs[i] : rhs[i]`
///
/// `lhs` and `rhs` must have the same type and length as `boolean_mask`.
/// `boolean_mask` must be of type BOOL8.
cudf_status_t cudf_copy_if_else_vv(cudf_column_view_handle lhs,
                                   cudf_column_view_handle rhs,
                                   cudf_column_view_handle boolean_mask,
                                   cudf_column_handle* out);

/// Conditionally select elements with explicit execution options.
cudf_status_t cudf_copy_if_else_vv_ex(cudf_column_view_handle lhs,
                                      cudf_column_view_handle rhs,
                                      cudf_column_view_handle boolean_mask,
                                      cudf_column_handle* out,
                                      const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_COPYING_H
