#ifndef CUDF_RUST_EVENT_H
#define CUDF_RUST_EVENT_H

#include "error.h"
#include "stream.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an owning CUDA event.
typedef struct cudf_event_handle_state* cudf_event_handle;

/// Create a new CUDA event.
cudf_status_t cudf_event_create(cudf_event_handle* out);

/// Destroy a CUDA event handle. Ownership is consumed on call, including on
/// failure; the handle must never be reused.
cudf_status_t cudf_event_destroy(cudf_event_handle handle);

/// Record an event on a CUDA stream.
cudf_status_t cudf_event_record(cudf_event_handle event, cudf_stream_handle stream);

/// Block the calling thread until the event has completed.
cudf_status_t cudf_event_synchronize(cudf_event_handle event);

/// Check whether the event has completed (1 ready, 0 not ready).
cudf_status_t cudf_event_is_ready(cudf_event_handle event, int32_t* out);

/// Make a CUDA stream wait on an event before proceeding.
cudf_status_t cudf_stream_wait_event(cudf_stream_handle stream, cudf_event_handle event);

/// Record a raw CUDA event on a raw CUDA stream.
///
/// Both pointers are borrowed `cudaEvent_t` / `cudaStream_t` values and do not
/// transfer ownership.
cudf_status_t cudf_raw_cuda_event_record(void* raw_event, void* raw_stream);

/// Make a raw CUDA stream wait on a raw CUDA event before proceeding.
///
/// Both pointers are borrowed `cudaStream_t` / `cudaEvent_t` values and do not
/// transfer ownership.
cudf_status_t cudf_raw_cuda_stream_wait_event(void* raw_stream, void* raw_event);

/// Borrow the underlying raw CUDA event pointer from an owning event handle.
///
/// This does not transfer ownership. `*out` is valid only while `handle`
/// remains alive.
cudf_status_t cudf_event_raw(cudf_event_handle handle, void** out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_EVENT_H
