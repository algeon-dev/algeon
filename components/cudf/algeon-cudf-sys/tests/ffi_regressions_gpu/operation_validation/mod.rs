use std::os::raw::c_void;

use crate::{
    helpers::*,
    support::{
        NO_SYNC_RECORDED, OwnedMemoryResource, OwnedStream, current_device, last_error_message, last_exec_mr_raw,
        last_exec_stream_value, last_exec_sync_stream_value, reset_exec_observation, stream_value,
    },
};

fn raw_handle<T>(handle: *mut T) -> *mut c_void {
    handle.cast()
}

fn raw_out_handle<T>(out: &mut *mut T) -> *mut *mut c_void {
    (out as *mut *mut T).cast()
}

fn raw_status(status: i32) -> cudf_sys::cudf_status_t {
    status as cudf_sys::cudf_status_t
}

fn required_exec_options(
    stream: &cudf_sys::BorrowedStream<'_>,
    mr: &cudf_sys::BorrowedMemoryResource<'_>,
) -> cudf_sys::cudf_exec_options_t {
    let mut options = cudf_sys::cudf_exec_options_t::defaults();
    options.device = current_device();
    options.flags = cudf_sys::CUDF_EXEC_FLAG_DEVICE_EXPLICIT;
    options.stream = stream.as_raw();
    options.mr = mr.as_raw();
    options
}

fn strict_exec_owners() -> (OwnedStream, OwnedMemoryResource, cudf_sys::cudf_exec_options_t) {
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    (stream, mr, exec)
}

mod column;
mod copying;
mod dictionary;
mod groupby;
mod join;
mod operators;
mod strings;
