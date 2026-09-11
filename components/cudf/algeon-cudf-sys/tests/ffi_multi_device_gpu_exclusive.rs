//! Switches the current CUDA device across two GPUs and asserts device-level
//! provenance and cross-device rejections, so it runs alone.
use std::{os::raw::c_char, ptr};

#[path = "ffi_regressions/helpers.rs"]
mod helpers;
mod support;

use helpers::{make_column_view, make_string_column};
use support::{
    CurrentDeviceRestore, OwnedMemoryResource, OwnedStream, current_device, last_error_message, lock_shim_state,
    required_exec_options, set_current_device, visible_device_count,
};

fn prepare_replace_with_backrefs(
    view: cudf_sys::cudf_column_view_handle,
    exec: &cudf_sys::cudf_exec_options_t,
) -> cudf_sys::cudf_prepared_replace_with_backrefs_handle {
    let pattern = r"([a-z]+)-(\d+)";
    let replacement = r"\2/\1";
    let mut prepared = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_prepare_replace_with_backrefs_ex(
            view,
            pattern.as_ptr() as *const c_char,
            pattern.len() as i32,
            replacement.as_ptr() as *const c_char,
            replacement.len() as i32,
            &mut prepared,
            exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!prepared.is_null());
    prepared
}

#[test]
fn prepared_replace_execute_rejects_device_mismatch_before_libcudf() {
    let _lock = lock_shim_state();
    if visible_device_count() < 2 {
        eprintln!("skipping prepared replace device-mismatch test; fewer than two CUDA devices are visible");
        return;
    }
    let _restore = CurrentDeviceRestore::new();
    set_current_device(0);
    let col = make_string_column(&["abc-123", "def-456"]);
    let view = make_column_view(col);
    let prepare_stream = OwnedStream::new();
    let prepare_mr = OwnedMemoryResource::new_test_stats();
    let prepare_exec = required_exec_options(&prepare_stream.borrowed(), &prepare_mr.borrowed());
    let baseline = prepare_mr.active_allocation_count();
    let prepared = prepare_replace_with_backrefs(view, &prepare_exec);

    set_current_device(1);
    {
        let execute_stream = OwnedStream::new();
        let execute_mr = OwnedMemoryResource::new_cuda();
        let execute_exec = required_exec_options(&execute_stream.borrowed(), &execute_mr.borrowed());
        let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
        let status =
            unsafe { cudf_sys::cudf_strings_execute_replace_with_backrefs_ex(view, prepared, &mut out, &execute_exec) };
        assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
        assert!(out.is_null());
        assert!(last_error_message().contains("device must match prepared replace-with-backrefs provenance"));
        assert_eq!(current_device(), 1, "provenance cleanup must restore the caller's active device");
    }

    set_current_device(0);
    assert_eq!(prepare_mr.active_allocation_count(), baseline);
    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_memory_resource_destroy_restores_current_device() {
    let _shim_state = lock_shim_state();
    let mut device_count = 0;
    let status = unsafe { cudf_sys::cudf_cuda_get_device_count(&mut device_count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    if device_count < 2 {
        return;
    }

    let _restore = CurrentDeviceRestore::new();
    set_current_device(0);

    let mut mr: cudf_sys::cudf_memory_resource_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_pool(1 << 20, 1 << 24, &mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());

    set_current_device(1);
    let before_destroy = current_device();
    assert_eq!(before_destroy, 1);

    let status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, 0) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(current_device(), before_destroy);
}

#[test]
fn native_manager_reservation_rejects_cross_device_stream() {
    let _shim_state = lock_shim_state();
    if visible_device_count() < 2 {
        eprintln!("skipping native cross-device stream reservation test; fewer than two CUDA devices are visible");
        return;
    }
    let _restore = CurrentDeviceRestore::new();

    set_current_device(0);
    let mut manager = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_manager_for_device(0, &mut manager) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!manager.is_null());

    set_current_device(1);
    let stream = OwnedStream::new();

    let mut reservation = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_memory_manager_attach_reservation(
            manager,
            stream.as_raw(),
            70_001,
            1024 * 1024,
            &mut reservation,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(reservation.is_null());
    assert!(
        last_error_message().contains("same CUDA device"),
        "expected same-device rejection, got {:?}",
        last_error_message()
    );

    drop(stream);
}

#[test]
fn native_query_domain_rejects_cross_device_stream() {
    let _shim_state = lock_shim_state();
    if visible_device_count() < 2 {
        eprintln!("skipping native cross-device query-domain test; fewer than two CUDA devices are visible");
        return;
    }
    let _restore = CurrentDeviceRestore::new();

    set_current_device(0);
    let mut manager = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_manager_for_device(0, &mut manager) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!manager.is_null());

    let mut domain = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_memory_manager_create_query_memory_domain(manager, 70_002, 1024 * 1024, 1024 * 1024, &mut domain)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!domain.is_null());

    set_current_device(1);
    let stream = OwnedStream::new();

    let mut reservation = ptr::null_mut();
    let mut resource = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_query_memory_domain_attach_stream(
            domain,
            stream.as_raw(),
            70_003,
            1024 * 1024,
            &mut reservation,
            &mut resource,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(reservation.is_null());
    assert!(resource.is_null());
    assert!(
        last_error_message().contains("same CUDA device"),
        "expected same-device rejection, got {:?}",
        last_error_message()
    );

    drop(stream);
    set_current_device(0);
    let status = unsafe { cudf_sys::cudf_query_memory_domain_destroy(domain) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
}
