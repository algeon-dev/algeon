#ifndef CUDF_RUST_STREAM_H
#define CUDF_RUST_STREAM_H

#include "error.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an owning CUDA stream.
typedef struct cudf_stream_handle_state* cudf_stream_handle;

/// Create a new CUDA stream.
///
/// Use this handle directly with `cudf_stream_synchronize` /
/// `cudf_stream_is_ready`, or attach it to a `cudf_exec_options_t` to direct
/// `_ex` APIs onto an explicit stream.
cudf_status_t cudf_stream_create(cudf_stream_handle* out);

/// Destroy a CUDA stream handle. Ownership is consumed on call, including on
/// failure; the handle must never be reused.
cudf_status_t cudf_stream_destroy(cudf_stream_handle handle);

/// Synchronize work already enqueued on a CUDA stream handle.
cudf_status_t cudf_stream_synchronize(cudf_stream_handle handle);

/// Synchronize work already enqueued on a raw borrowed CUDA stream pointer.
///
/// `raw_stream` must be a valid borrowed `cudaStream_t` value. This does not
/// take ownership and is intended for compatibility layers that carry raw CUDA
/// stream identity rather than a `cudf_stream_handle`.
cudf_status_t cudf_raw_cuda_stream_synchronize(void* raw_stream);

/// Check whether work already enqueued on a CUDA stream handle has completed
/// (1 ready, 0 not ready).
cudf_status_t cudf_stream_is_ready(cudf_stream_handle handle, int32_t* out);

/// Borrow the underlying raw CUDA stream pointer from an owning stream handle.
///
/// This does not transfer ownership. `*out` is valid only while `handle`
/// remains alive.
cudf_status_t cudf_stream_raw(cudf_stream_handle handle, void** out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_STREAM_H
