#ifndef CUDF_RUST_COPYING_H
#define CUDF_RUST_COPYING_H

#include "error.h"
#include "table.h"
#include "column.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Out-of-bounds policy for gather operations.
typedef enum {
  CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY = 0,
  CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK = 1,
} cudf_out_of_bounds_policy_t;

/// Gather rows from a table by index.
/// `gather_map` must be a non-nullable INT32 column of row indices.
cudf_status_t cudf_gather(cudf_table_view_handle input,
                          cudf_column_view_handle gather_map,
                          cudf_out_of_bounds_policy_t bounds_policy,
                          cudf_table_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_COPYING_H
