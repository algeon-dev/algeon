use std::{os::raw::c_void, ptr};

use crate::support::{OwnedEvent, OwnedStream, current_device, lock_shim_state};

#[test]
fn test_stream_is_ready_null_out_rejected() {
    let _shim_state = lock_shim_state();
    let mut stream: cudf_sys::cudf_stream_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_stream_create(&mut stream) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!stream.is_null());

    let st = unsafe { cudf_sys::cudf_stream_is_ready(stream, ptr::null_mut()) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);

    unsafe { cudf_sys::cudf_stream_destroy(stream) };
}

#[test]
fn test_stream_create_is_ready_and_synchronize_smoke() {
    let _shim_state = lock_shim_state();
    let mut stream: cudf_sys::cudf_stream_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_stream_create(&mut stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!stream.is_null());

    let mut ready = 0;
    let status = unsafe { cudf_sys::cudf_stream_is_ready(stream, &mut ready) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(ready, 1);

    let status = unsafe { cudf_sys::cudf_stream_synchronize(stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    unsafe { cudf_sys::cudf_stream_destroy(stream) };
}

#[test]
fn test_event_create_null_out_rejected() {
    let _shim_state = lock_shim_state();
    let status = unsafe { cudf_sys::cudf_event_create(ptr::null_mut()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
}

#[test]
fn test_event_create_record_wait_and_synchronize_smoke() {
    let _shim_state = lock_shim_state();
    unsafe { cudf_sys::test_support::cudf_test_event_reset_observation() };

    let mut stream: cudf_sys::cudf_stream_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_stream_create(&mut stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!stream.is_null());

    let mut event: cudf_sys::cudf_event_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_event_create(&mut event) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!event.is_null());

    let status = unsafe { cudf_sys::cudf_event_record(event, stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let mut ready = 0;
    let status = unsafe { cudf_sys::cudf_event_is_ready(event, &mut ready) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(ready == 0 || ready == 1);

    let status = unsafe { cudf_sys::cudf_stream_wait_event(stream, event) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let status = unsafe { cudf_sys::cudf_event_synchronize(event) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let mut synchronize_calls = -1;
    let status = unsafe { cudf_sys::test_support::cudf_test_event_synchronize_call_count(&mut synchronize_calls) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(synchronize_calls, 1);

    let status = unsafe { cudf_sys::cudf_event_is_ready(event, &mut ready) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(ready, 1);

    let mut raw: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_event_raw(event, &mut raw) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!raw.is_null());

    unsafe {
        cudf_sys::cudf_event_destroy(event);
        cudf_sys::cudf_stream_destroy(stream);
    }
}

#[test]
fn test_raw_cuda_event_record_and_wait_smoke() {
    let _shim_state = lock_shim_state();
    let stream = OwnedStream::new();
    let event = OwnedEvent::new();

    let raw_stream = stream.raw_cuda_stream();
    let raw_event = event.raw_cuda_event();

    let status = unsafe { cudf_sys::cudf_raw_cuda_event_record(raw_event, raw_stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let status = unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(raw_stream, raw_event) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let status = unsafe { cudf_sys::cudf_event_synchronize(event.as_raw()) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
}

#[test]
fn test_memory_resource_pool_create_destroy_smoke() {
    let _shim_state = lock_shim_state();
    let device = current_device();
    let mut mr: cudf_sys::cudf_memory_resource_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_pool(1 << 20, 1 << 24, &mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());

    let status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
}

#[test]
fn test_memory_resource_cuda_create_destroy_smoke() {
    let _shim_state = lock_shim_state();
    let device = current_device();
    let mut mr: cudf_sys::cudf_memory_resource_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());

    let status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
}
