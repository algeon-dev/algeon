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

/// Bit 0 of `flags`: when set, `device` was explicitly provided by the caller
/// and the shim must switch to it before resolving stream/MR defaults.
#define CUDF_EXEC_FLAG_DEVICE_EXPLICIT 1u

/// Versioned execution options for compute and host-materialization APIs.
///
/// Set `struct_size` to `sizeof(cudf_exec_options_t)`.
///
/// Passing NULL for the options pointer, or NULL for `stream` / `mr`, selects
/// cuDF's default stream and current device memory resource.
///
/// `device`:  CUDA device ordinal to execute on.  When
///            `CUDF_EXEC_FLAG_DEVICE_EXPLICIT` is set in `flags`, the shim
///            switches to this device before resolving stream/MR defaults.
///            When the flag is clear, `device` is ignored and the calling
///            thread's current device is used.
///
/// `mr` must outlive any outputs allocated through it.
///
/// `reserved` must be zero in this version.
typedef struct cudf_exec_options_t {
  size_t struct_size;
  int32_t device;
  uint32_t flags;
  cudf_stream_handle stream;
  cudf_memory_resource_handle mr;
  uint64_t reserved[4];
} cudf_exec_options_t;

#define CUDF_EXEC_OPTIONS_INITIALIZER \
  {sizeof(cudf_exec_options_t), -1, 0u, NULL, NULL, {0u, 0u, 0u, 0u}}

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_EXECUTION_CONTEXT_H
