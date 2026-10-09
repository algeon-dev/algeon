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

/// When bit 0 of `flags` is set, `device` was explicitly provided by the caller
/// and the shim must switch to it before entering libcudf.
#define CUDF_EXEC_FLAG_DEVICE_EXPLICIT 1u

/// When bit 1 of `flags` is set, `reserved[0]` bounds the read and decompression
/// scratch of Parquet reads, in bytes, which row-group selection alone cannot
/// bound. The rows returned are unchanged. The limit is a hint. A single row
/// group that does not fit within it is still loaded.
#define CUDF_EXEC_FLAG_PARQUET_PASS_READ_LIMIT 2u

/// Versioned execution options for compute and host-materialization APIs.
///
/// Set `struct_size` to `sizeof(cudf_exec_options_t)`.
///
/// Execution-sensitive production entry points require a non-NULL options
/// pointer, `CUDF_EXEC_FLAG_DEVICE_EXPLICIT`, a non-NULL `stream`, and a
/// non-NULL `mr`. Passing NULL for the options pointer, clearing the explicit
/// device flag, or leaving `stream` / `mr` NULL is only accepted by narrow
/// permissive diagnostics and legacy regression hooks.
///
/// `device`:  CUDA device ordinal to execute on.  When
///            `CUDF_EXEC_FLAG_DEVICE_EXPLICIT` is set in `flags`, the shim
///            switches to this device before entering libcudf. Production
///            operation paths reject options with the flag clear.
///
/// `mr` must outlive any outputs allocated through it.
///
/// `reserved[0]` carries the Parquet pass read limit in bytes when
/// `CUDF_EXEC_FLAG_PARQUET_PASS_READ_LIMIT` is set in `flags`; it must be zero
/// otherwise. The remaining `reserved` entries must be zero in this version.
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
