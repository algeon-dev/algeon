mod support;

use std::{ffi::c_void, ptr};

use support::{CurrentDeviceResourceRestore, current_device_resource_raw, lock_shim_state};

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
    let stats = cudf_sys::MemoryResource::new_test_stats().expect("failed to create stats memory resource");
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);

    cudf_sys::test_set_current_device_resource(&stats).expect("failed to install current test allocator");

    let installed_raw = current_device_resource_raw();
    assert_ne!(installed_raw, previous_raw, "test allocator should replace the previous current allocator");

    let before = stats.test_allocation_count().expect("failed to read allocation count before allocation");
    let col = make_i32_column(&[1, 2, 3, 4]);
    let after = stats.test_allocation_count().expect("failed to read allocation count after allocation");

    assert!(
        after > before,
        "allocating a column through the current allocator should increase the stats allocation count"
    );

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

#[test]
fn raw_memory_resource_accessor_returns_underlying_device_resource_pointer() {
    let mr = cudf_sys::MemoryResource::new_cuda().expect("failed to create CUDA memory resource");
    let raw = mr.raw_device_resource().expect("failed to resolve underlying device memory resource");
    assert!(!raw.is_null(), "underlying device memory resource pointer must be non-null");
}

#[test]
fn raw_stream_accessor_returns_underlying_cuda_stream_pointer() {
    let stream = cudf_sys::Stream::new().expect("failed to create CUDA stream");
    let raw = stream.raw_cuda_stream().expect("failed to resolve underlying CUDA stream pointer");
    assert!(!raw.is_null(), "underlying CUDA stream pointer must be non-null");
}
