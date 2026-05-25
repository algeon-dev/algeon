#ifndef CUDF_RUST_FILTER_H
#define CUDF_RUST_FILTER_H

#include "error.h"
#include "table.h"
#include "column.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Apply a boolean mask to a table. Rows where the mask is `true` are kept;
/// rows where the mask is `false` or null are dropped.
/// The mask must be a BOOL8 column with the same number of rows as the table.
cudf_status_t cudf_apply_boolean_mask(cudf_table_view_handle input,
                                      cudf_column_view_handle mask,
                                      cudf_table_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_FILTER_H
