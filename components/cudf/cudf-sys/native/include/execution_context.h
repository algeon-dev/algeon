#ifndef CUDF_RUST_EXECUTION_CONTEXT_H
#define CUDF_RUST_EXECUTION_CONTEXT_H

#include "error.h"
#include "memory_resource.h"
#include "stream.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Versioned execution options for compute and host-materialization APIs.
///
/// Set `struct_size` to `sizeof(cudf_exec_options_t)`.
///
/// Passing NULL for the options pointer, or NULL for `stream` / `mr`, selects
/// cuDF's default stream and current device memory resource.
///
/// `mr` must outlive any outputs allocated through it.
///
/// `flags` and `reserved` must be zero in this version.
typedef struct cudf_exec_options_t {
  size_t struct_size;
  cudf_stream_handle stream;
  cudf_memory_resource_handle mr;
  uint64_t flags;
  uint64_t reserved[4];
} cudf_exec_options_t;

#define CUDF_EXEC_OPTIONS_INITIALIZER \
  { sizeof(cudf_exec_options_t), NULL, NULL, 0u, {0u, 0u, 0u, 0u} }

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_EXECUTION_CONTEXT_H
