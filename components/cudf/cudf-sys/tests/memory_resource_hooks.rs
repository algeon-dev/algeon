mod support;

use std::{ffi::c_void, ptr};

use support::{CurrentDeviceResourceRestore, OwnedMemoryResource, OwnedStream, current_device_resource_raw, lock_shim_state};

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
