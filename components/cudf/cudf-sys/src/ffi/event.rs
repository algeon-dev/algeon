use std::os::raw::c_void;

use crate::{cudf_event_handle, cudf_status_t, cudf_stream_handle};

extern "C" {
    // -- Event ---------------------------------------------------------------

    /// Create an owned CUDA event handle.
    pub fn cudf_event_create(out: *mut cudf_event_handle) -> cudf_status_t;

    /// Destroy a CUDA event created by [`cudf_event_create`].
    pub fn cudf_event_destroy(handle: cudf_event_handle);

    /// Record an event on a CUDA stream handle.
    pub fn cudf_event_record(event: cudf_event_handle, stream: cudf_stream_handle) -> cudf_status_t;

    /// Block until a recorded event completes.
    pub fn cudf_event_synchronize(event: cudf_event_handle) -> cudf_status_t;

    /// Report whether a recorded event has completed.
    pub fn cudf_event_is_ready(event: cudf_event_handle, out: *mut i32) -> cudf_status_t;

    /// Make `stream` wait for `event` before future work proceeds.
    pub fn cudf_stream_wait_event(stream: cudf_stream_handle, event: cudf_event_handle) -> cudf_status_t;

    /// Record a borrowed raw CUDA event on a borrowed raw CUDA stream.
    pub fn cudf_raw_cuda_event_record(raw_event: *mut c_void, raw_stream: *mut c_void) -> cudf_status_t;

    /// Make a borrowed raw CUDA stream wait on a borrowed raw CUDA event.
    pub fn cudf_raw_cuda_stream_wait_event(raw_stream: *mut c_void, raw_event: *mut c_void) -> cudf_status_t;

    /// Borrow the underlying raw CUDA event pointer from `handle`.
    ///
    /// This does not transfer ownership. The returned pointer is valid only
    /// while `handle` remains alive.
    pub fn cudf_event_raw(handle: cudf_event_handle, out: *mut *mut c_void) -> cudf_status_t;

    /// Reset CUDA event observation hooks used by integration tests.
    pub fn cudf_test_event_reset_observation();

    /// Make subsequent observed `cudaEventSynchronize` calls block until released.
    pub fn cudf_test_event_block_synchronize();

    /// Release any test-only block applied to observed `cudaEventSynchronize` calls.
    pub fn cudf_test_event_unblock_synchronize();

    /// Force subsequent observed `cudaEventSynchronize` calls to fail.
    pub fn cudf_test_event_fail_synchronize();

    /// Clear any forced failure applied to observed `cudaEventSynchronize` calls.
    pub fn cudf_test_event_allow_synchronize();

    /// Read the number of observed `cudaEventSynchronize` calls.
    pub fn cudf_test_event_synchronize_call_count(out: *mut i32) -> cudf_status_t;

    /// Read the number of observed `cudaEventSynchronize` calls for one event handle pointer.
    pub fn cudf_test_event_synchronize_call_count_for_event_handle_raw(
        event_handle_raw: usize,
        out: *mut i32,
    ) -> cudf_status_t;

    /// Read the number of observed `cudaStreamWaitEvent` calls.
    pub fn cudf_test_event_stream_wait_call_count(out: *mut i32) -> cudf_status_t;

    /// Read the number of observed `cudaStreamWaitEvent` calls for one raw stream.
    pub fn cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw: usize, out: *mut i32) -> cudf_status_t;

    /// Read the number of observed `cudaStreamWaitEvent` calls for one raw event.
    pub fn cudf_test_event_stream_wait_call_count_for_event_raw(event_raw: usize, out: *mut i32) -> cudf_status_t;

    /// Read the last raw CUDA stream pointer passed to `cudaStreamWaitEvent`.
    pub fn cudf_test_event_last_wait_stream_raw(out: *mut usize) -> cudf_status_t;

    /// Read the last raw CUDA event pointer passed to `cudaStreamWaitEvent`.
    pub fn cudf_test_event_last_wait_event_raw(out: *mut usize) -> cudf_status_t;

    /// Read the number of observed raw `cudaEventRecord` calls.
    pub fn cudf_test_event_raw_record_call_count(out: *mut i32) -> cudf_status_t;

    /// Read the number of observed raw `cudaEventRecord` calls for one raw stream.
    pub fn cudf_test_event_raw_record_call_count_for_stream_raw(stream_raw: usize, out: *mut i32) -> cudf_status_t;

    /// Read the last raw CUDA stream pointer passed to raw `cudaEventRecord`.
    pub fn cudf_test_event_last_raw_record_stream_raw(out: *mut usize) -> cudf_status_t;

    /// Read the last raw CUDA event pointer passed to raw `cudaEventRecord`.
    pub fn cudf_test_event_last_raw_record_event_raw(out: *mut usize) -> cudf_status_t;
}
