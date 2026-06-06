#ifndef CUDF_RUST_HASHING_H
#define CUDF_RUST_HASHING_H

#include "column.h"
#include "execution_context.h"
#include "error.h"
#include "table.h"
#include "types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Compute MurmurHash3 32-bit hash of each row in a table.
///
/// Returns an owning UINT32 column with one hash value per row.
/// The caller must free the result with `cudf_column_destroy`.
cudf_status_t cudf_hashing_murmurhash3_x86_32(cudf_table_view_handle input,
                                               uint32_t seed,
                                               cudf_column_handle* out);

/// Compute MurmurHash3 32-bit hash of each row in a table with explicit
/// execution context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_hashing_murmurhash3_x86_32_ex(
    cudf_table_view_handle input,
    uint32_t seed,
    cudf_column_handle* out,
    const cudf_exec_options_t* options);

/// Compute XXHash 64-bit hash of each row in a table.
///
/// Returns an owning UINT64 column with one hash value per row.
/// The caller must free the result with `cudf_column_destroy`.
cudf_status_t cudf_hashing_xxhash_64(cudf_table_view_handle input,
                                     uint64_t seed,
                                     cudf_column_handle* out);

/// Compute XXHash 64-bit hash of each row in a table with explicit execution
/// context.
///
/// Passing NULL for `options`, or leaving `options->stream` / `options->mr`
/// NULL, uses cuDF's default stream and current device memory resource.
/// When `options->mr` is non-NULL, it must outlive the returned column.
cudf_status_t cudf_hashing_xxhash_64_ex(cudf_table_view_handle input,
                                        uint64_t seed,
                                        cudf_column_handle* out,
                                        const cudf_exec_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_HASHING_H
