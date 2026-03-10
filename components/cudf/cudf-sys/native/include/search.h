#ifndef CUDF_RUST_SEARCH_H
#define CUDF_RUST_SEARCH_H

#include "error.h"
#include "column.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Check if elements of `needles` exist in `haystack`.
/// Returns a BOOL8 column with the same size as `needles`.
cudf_status_t cudf_contains(cudf_column_view_handle haystack,
                            cudf_column_view_handle needles,
                            cudf_column_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_SEARCH_H
