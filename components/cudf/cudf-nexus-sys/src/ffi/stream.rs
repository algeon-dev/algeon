use std::os::raw::c_void;

use crate::{cudf_status_t, cudf_stream_handle};

unsafe extern "C" {
    // -- Stream --------------------------------------------------------------

    /// Create an owned CUDA stream handle.
    ///
    /// Attach this handle to [`crate::ExecutionContext`] or
    /// [`crate::cudf_exec_options_t`] when calling `_ex` APIs to direct work
    /// onto an explicit stream.
    pub fn cudf_stream_create(out: *mut cudf_stream_handle) -> cudf_status_t;

    /// Destroy a CUDA stream created by [`cudf_stream_create`].
    pub fn cudf_stream_destroy(handle: cudf_stream_handle) -> cudf_status_t;

    /// Block until work already enqueued on `handle` completes.
    pub fn cudf_stream_synchronize(handle: cudf_stream_handle) -> cudf_status_t;

    /// Block until work already enqueued on one borrowed raw CUDA stream completes.
    pub fn cudf_raw_cuda_stream_synchronize(raw_stream: *mut c_void) -> cudf_status_t;

    /// Report whether work already enqueued on `handle` has completed.
    pub fn cudf_stream_is_ready(handle: cudf_stream_handle, out: *mut i32) -> cudf_status_t;

    /// Borrow the underlying raw CUDA stream pointer from `handle`.
    ///
    /// This does not transfer ownership. The returned pointer is valid only
    /// while `handle` remains alive.
    pub fn cudf_stream_raw(handle: cudf_stream_handle, out: *mut *mut c_void) -> cudf_status_t;
}
