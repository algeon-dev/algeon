#ifndef CUDF_RUST_COPYING_H
#define CUDF_RUST_COPYING_H

#include "error.h"
#include "table.h"
#include "column.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Out-of-bounds policy for gather operations.
/// 0 = NULLIFY (out-of-bounds indices become null)
/// 1 = DONT_CHECK (no bounds checking, undefined on OOB)

/// Gather rows from a table by index.
/// `gather_map` must be a non-nullable INT32 column of row indices.
cudf_status_t cudf_gather(cudf_table_view_handle input,
                          cudf_column_view_handle gather_map,
                          int bounds_policy,
                          cudf_table_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_COPYING_H
