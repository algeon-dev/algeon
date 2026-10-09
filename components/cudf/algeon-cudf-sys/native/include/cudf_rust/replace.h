#ifndef CUDF_RUST_REPLACE_H
#define CUDF_RUST_REPLACE_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "scalar.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Replace null values in a column with a scalar using an explicit execution
/// context.
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_replace_nulls_scalar_ex(cudf_column_view_handle input,
                                           cudf_scalar_handle replacement,
                                           cudf_column_handle* out,
                                           cudf_exec_options_t const* options);

/// Replace null values in a column with corresponding values from another
/// column.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_replace_nulls_column_ex(cudf_column_view_handle input,
                                           cudf_column_view_handle replacement,
                                           cudf_column_handle* out,
                                           cudf_exec_options_t const* options);

/// Replace null values using a fill policy with an explicit execution context.
/// `policy`: 0 = PRECEDING (forward fill), 1 = FOLLOWING (backward fill).
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_replace_nulls_policy_ex(cudf_column_view_handle input,
                                           int32_t policy,
                                           cudf_column_handle* out,
                                           cudf_exec_options_t const* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_REPLACE_H
