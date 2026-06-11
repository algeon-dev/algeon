#ifndef CUDF_RUST_SEARCH_H
#define CUDF_RUST_SEARCH_H

#include "column.h"
#include "error.h"
#include "execution_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Check if elements of `needles` exist in `haystack`.
/// Returns a BOOL8 column with the same size as `needles`.
cudf_status_t cudf_contains(cudf_column_view_handle haystack,
                            cudf_column_view_handle needles,
                            cudf_column_handle* out);

/// Check if elements of `needles` exist in `haystack` with explicit execution
/// context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_contains_ex(cudf_column_view_handle haystack,
                               cudf_column_view_handle needles,
                               cudf_column_handle* out,
                               const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_SEARCH_H
