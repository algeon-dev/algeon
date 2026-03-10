#ifndef CUDF_RUST_REPLACE_H
#define CUDF_RUST_REPLACE_H

#include "error.h"
#include "column.h"
#include "scalar.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Replace null values in a column with a scalar.
/// The scalar type must match the column type.
cudf_status_t cudf_replace_nulls_scalar(cudf_column_view_handle input,
                                        cudf_scalar_handle replacement,
                                        cudf_column_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_REPLACE_H
