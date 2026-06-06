#ifndef CUDF_RUST_REPLACE_H
#define CUDF_RUST_REPLACE_H

#include "error.h"
#include "column.h"
#include "execution_context.h"
#include "scalar.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Replace null values in a column with a scalar.
/// The scalar type must match the column type.
cudf_status_t cudf_replace_nulls_scalar(cudf_column_view_handle input,
                                        cudf_scalar_handle replacement,
                                        cudf_column_handle* out);

/// Replace null values in a column with a scalar using an explicit execution
/// context. Null `options` preserve cuDF defaults.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_replace_nulls_scalar_ex(cudf_column_view_handle input,
                                           cudf_scalar_handle replacement,
                                           cudf_column_handle* out,
                                           const cudf_exec_options_t* options);

/// Replace null values in a column with corresponding values from another column.
/// Both columns must have the same type and length.
cudf_status_t cudf_replace_nulls_column(cudf_column_view_handle input,
                                        cudf_column_view_handle replacement,
                                        cudf_column_handle* out);

/// Replace null values in a column with corresponding values from another
/// column using an explicit execution context. Null `options` preserve cuDF
/// defaults.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_replace_nulls_column_ex(cudf_column_view_handle input,
                                           cudf_column_view_handle replacement,
                                           cudf_column_handle* out,
                                           const cudf_exec_options_t* options);

/// Replace null values using a fill policy (forward or backward fill).
/// `policy`: 0 = PRECEDING (forward fill), 1 = FOLLOWING (backward fill).
cudf_status_t cudf_replace_nulls_policy(cudf_column_view_handle input,
                                        int32_t policy,
                                        cudf_column_handle* out);

/// Replace null values using a fill policy with an explicit execution context.
/// `policy`: 0 = PRECEDING (forward fill), 1 = FOLLOWING (backward fill).
/// Null `options` preserve cuDF defaults.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_replace_nulls_policy_ex(cudf_column_view_handle input,
                                           int32_t policy,
                                           cudf_column_handle* out,
                                           const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_REPLACE_H
