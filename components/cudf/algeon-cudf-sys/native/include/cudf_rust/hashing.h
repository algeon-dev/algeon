#ifndef CUDF_RUST_HASHING_H
#define CUDF_RUST_HASHING_H

#include "column.h"
#include "error.h"
#include "execution_context.h"
#include "table.h"
#include "types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Compute MurmurHash3 32-bit hash of each row in a table with explicit
/// execution context.
///
/// `options` must be non-NULL and must carry an explicit device, stream, and
/// memory resource. `options->mr` must outlive the returned column.
cudf_status_t cudf_hashing_murmurhash3_x86_32_ex(cudf_table_view_handle input,
                                                 uint32_t seed,
                                                 cudf_column_handle* out,
                                                 cudf_exec_options_t const* options);

/// Compute XXHash 64-bit hash of each row in a table with explicit execution
/// context.
///
/// `options` must be non-NULL and must carry an explicit device, stream, and
/// memory resource. `options->mr` must outlive the returned column.
cudf_status_t cudf_hashing_xxhash_64_ex(cudf_table_view_handle input,
                                        uint64_t seed,
                                        cudf_column_handle* out,
                                        cudf_exec_options_t const* options);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_HASHING_H
