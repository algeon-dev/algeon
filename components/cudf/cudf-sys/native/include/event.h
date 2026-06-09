#ifndef CUDF_RUST_EVENT_H
#define CUDF_RUST_EVENT_H

#include "error.h"
#include "stream.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to an owning CUDA event.
typedef void* cudf_event_handle;

/// Create a new CUDA event.
cudf_status_t cudf_event_create(cudf_event_handle* out);

/// Destroy a CUDA event handle.
void cudf_event_destroy(cudf_event_handle handle);

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

/// Reset CUDA event observation hooks used by integration tests.
void cudf_test_event_reset_observation();

/// Make subsequent `cudaEventSynchronize` shim calls block until explicitly released.
void cudf_test_event_block_synchronize();

/// Release any test-only block applied to `cudaEventSynchronize` shim calls.
void cudf_test_event_unblock_synchronize();

/// Force subsequent observed `cudaEventSynchronize` calls to fail.
void cudf_test_event_fail_synchronize();

/// Clear any forced failure applied to observed `cudaEventSynchronize` calls.
void cudf_test_event_allow_synchronize();

/// Read the number of `cudaEventSynchronize` calls observed through the shim.
cudf_status_t cudf_test_event_synchronize_call_count(int32_t* out);

/// Read the number of `cudaStreamWaitEvent` calls observed through the shim.
cudf_status_t cudf_test_event_stream_wait_call_count(int32_t* out);

/// Read the number of `cudaStreamWaitEvent` calls observed for one raw stream.
cudf_status_t cudf_test_event_stream_wait_call_count_for_stream_raw(uintptr_t stream_raw, int32_t* out);

/// Read the number of `cudaStreamWaitEvent` calls observed for one raw event.
cudf_status_t cudf_test_event_stream_wait_call_count_for_event_raw(uintptr_t event_raw, int32_t* out);

/// Read the last raw CUDA stream pointer passed to `cudaStreamWaitEvent`.
cudf_status_t cudf_test_event_last_wait_stream_raw(uintptr_t* out);

/// Read the last raw CUDA event pointer passed to `cudaStreamWaitEvent`.
cudf_status_t cudf_test_event_last_wait_event_raw(uintptr_t* out);

/// Read the number of raw `cudaEventRecord` calls observed through the shim.
cudf_status_t cudf_test_event_raw_record_call_count(int32_t* out);

/// Read the number of raw `cudaEventRecord` calls observed for one raw stream.
cudf_status_t cudf_test_event_raw_record_call_count_for_stream_raw(uintptr_t stream_raw, int32_t* out);

/// Read the last raw CUDA stream pointer passed to `cudaEventRecord` via the raw shim.
cudf_status_t cudf_test_event_last_raw_record_stream_raw(uintptr_t* out);

/// Read the last raw CUDA event pointer passed to `cudaEventRecord` via the raw shim.
cudf_status_t cudf_test_event_last_raw_record_event_raw(uintptr_t* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_EVENT_H
