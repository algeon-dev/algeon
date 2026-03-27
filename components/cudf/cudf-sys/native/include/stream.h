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
///
/// This handle is currently a standalone control surface for
/// `cudf_stream_synchronize` and `cudf_stream_is_ready` only. Other compute
/// APIs in this shim do not accept a stream handle and continue to execute on
/// cuDF's default stream.
cudf_status_t cudf_stream_create(cudf_stream_handle* out);

/// Destroy a CUDA stream handle.
void cudf_stream_destroy(cudf_stream_handle handle);

/// Synchronize work already enqueued on a CUDA stream handle.
cudf_status_t cudf_stream_synchronize(cudf_stream_handle handle);

/// Check whether work already enqueued on a CUDA stream handle has completed
/// (1 ready, 0 not ready).
cudf_status_t cudf_stream_is_ready(cudf_stream_handle handle, int32_t* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STREAM_H
