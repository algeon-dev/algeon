#ifndef CUDF_RUST_STREAM_H
#define CUDF_RUST_STREAM_H

#include "error.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an owning CUDA stream.
typedef void* cudf_stream_handle;

/// Create a new CUDA stream.
cudf_status_t cudf_stream_create(cudf_stream_handle* out);

/// Destroy a CUDA stream handle.
void cudf_stream_destroy(cudf_stream_handle handle);

/// Synchronize a CUDA stream.
cudf_status_t cudf_stream_synchronize(cudf_stream_handle handle);

/// Check whether stream work has completed (1 ready, 0 not ready).
cudf_status_t cudf_stream_is_ready(cudf_stream_handle handle, int32_t* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STREAM_H
