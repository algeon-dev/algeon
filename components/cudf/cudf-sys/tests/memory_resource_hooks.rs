mod support;

use std::{ffi::c_void, ptr};

use support::{
    CurrentDeviceResourceRestore, OwnedMemoryResource, OwnedStream, current_device, current_device_resource_raw,
    lock_shim_state,
};

fn make_i32_column(data: &[i32]) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn create_cuda_memory_resource() -> *mut c_void {
    let mut mr: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());
    mr
}

fn create_pool_memory_resource() -> *mut c_void {
    let mut mr: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_pool(1 << 20, 1 << 24, &mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());
    mr
}

#[test]
fn stats_memory_resource_can_be_installed_as_current_allocator() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let stats = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let borrowed = stats.borrowed();

    cudf_sys::test_set_current_device_resource(&borrowed).expect("failed to install current test allocator");

    let installed_raw = current_device_resource_raw();
    assert_ne!(installed_raw, previous_raw, "test allocator should replace the previous current allocator");

    let before = stats.allocation_count();
    let col = make_i32_column(&[1, 2, 3, 4]);
    let after = stats.allocation_count();

    assert!(
        after > before,
        "allocating a column through the current allocator should increase the stats allocation count"
    );

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

#[test]
fn destroying_current_memory_resource_returns_invalid_operation_and_preserves_current_resource() {
    let _shim_state = lock_shim_state();
    let device = current_device();
    let previous_raw = current_device_resource_raw();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);

    for (name, create) in [
        ("cuda", create_cuda_memory_resource as fn() -> *mut c_void),
        ("pool", create_pool_memory_resource as fn() -> *mut c_void),
    ] {
        let mr = create();

        let mut installed_raw: *mut c_void = ptr::null_mut();
        let status = unsafe { cudf_sys::cudf_memory_resource_raw(mr, &mut installed_raw) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        assert!(!installed_raw.is_null());

        let status = unsafe { cudf_sys::cudf_test_memory_resource_set_current(mr) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        assert_eq!(
            current_device_resource_raw(),
            installed_raw,
            "{name} test allocator should be installed as the current device resource before destroy"
        );

        let destroy_status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
        let current_after_destroy = current_device_resource_raw();

        unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource before cleanup");

        if destroy_status == cudf_sys::CUDF_ERROR_INVALID_OPERATION {
            let cleanup_status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
            assert_eq!(cleanup_status, cudf_sys::CUDF_SUCCESS);
        }

        assert_eq!(destroy_status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
        assert_eq!(
            current_after_destroy, installed_raw,
            "failed destroy of the current {name} device resource must not rewrite global current resource state"
        );
    }
}

#[test]
fn destroying_non_current_memory_resource_still_succeeds_after_current_destroy_rejection() {
    let _shim_state = lock_shim_state();
    let device = current_device();
    let previous_raw = current_device_resource_raw();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);

    let mr = create_cuda_memory_resource();

    let status = unsafe { cudf_sys::cudf_test_memory_resource_set_current(mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let destroy_status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
    assert_eq!(destroy_status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource before cleanup");

    let cleanup_status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
    assert_eq!(cleanup_status, cudf_sys::CUDF_SUCCESS);

    let fresh_mr = create_cuda_memory_resource();
    let fresh_destroy_status = unsafe { cudf_sys::cudf_memory_resource_destroy(fresh_mr, device) };
    assert_eq!(fresh_destroy_status, cudf_sys::CUDF_SUCCESS);
}

#[test]
fn raw_memory_resource_accessor_returns_underlying_device_resource_pointer() {
    let mr = OwnedMemoryResource::new_cuda();
    let raw = mr.raw_device_resource();
    assert!(!raw.is_null(), "underlying device memory resource pointer must be non-null");
}

#[test]
fn raw_stream_accessor_returns_underlying_cuda_stream_pointer() {
    let stream = OwnedStream::new();
    let raw = stream.raw_cuda_stream();
    assert!(!raw.is_null(), "underlying CUDA stream pointer must be non-null");
}
