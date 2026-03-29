use std::{ffi::c_void, ptr};

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

fn current_device_resource_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    out
}

struct CurrentDeviceResourceRestore {
    previous_raw: *mut c_void,
}

impl Drop for CurrentDeviceResourceRestore {
    fn drop(&mut self) {
        unsafe { cudf_sys::test_set_current_device_resource_raw(self.previous_raw) }
            .expect("failed to restore previous current device resource");
    }
}

#[test]
fn stats_memory_resource_can_be_installed_as_current_allocator() {
    let previous_raw = current_device_resource_raw();
    let stats = cudf_sys::MemoryResource::new_test_stats().expect("failed to create stats memory resource");
    let _restore = CurrentDeviceResourceRestore { previous_raw };

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
