#ifndef CUDF_RUST_SEARCH_H
#define CUDF_RUST_SEARCH_H

#include "column.h"
#include "error.h"
#include "execution_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Check if elements of `needles` exist in `haystack` with explicit execution
/// context.
///
/// `options` must be non-NULL, name an explicit valid device, provide a
/// non-NULL stream, and provide a non-NULL memory resource. `options->mr` must
/// outlive the returned column.
cudf_status_t cudf_contains_ex(cudf_column_view_handle haystack,
                               cudf_column_view_handle needles,
                               cudf_column_handle* out,
                               cudf_exec_options_t const* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_SEARCH_H
