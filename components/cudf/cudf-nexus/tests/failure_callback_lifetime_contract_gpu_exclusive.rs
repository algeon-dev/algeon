use std::{
    ffi::c_void,
    ptr,
    sync::{Arc, Weak},
};

use cudf::{DeviceId, DeviceMemoryResource};

fn current_device_resource_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::test_support::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    out
}

fn set_current_device_resource(mr: &cudf_sys::BorrowedMemoryResource<'_>) -> Result<(), cudf_sys::cudf_status_t> {
    let status = unsafe { cudf_sys::test_support::cudf_test_memory_resource_set_current(mr.as_raw()) };
    if status == cudf_sys::CUDF_SUCCESS { Ok(()) } else { Err(status) }
}

unsafe fn set_current_device_resource_raw(raw_mr: *mut c_void) -> Result<(), cudf_sys::cudf_status_t> {
    let status = unsafe { cudf_sys::test_support::cudf_test_memory_resource_set_current_raw(raw_mr) };
    if status == cudf_sys::CUDF_SUCCESS { Ok(()) } else { Err(status) }
}

struct CurrentDeviceResourceRestore {
    previous_raw: *mut c_void,
}

impl Drop for CurrentDeviceResourceRestore {
    fn drop(&mut self) {
        unsafe { set_current_device_resource_raw(self.previous_raw) }
            .expect("failed to restore previous current device resource");
    }
}

#[test]
fn failure_callback_state_survives_destroy_rejection_while_resource_is_current() {
    let previous_raw = current_device_resource_raw();
    let _restore = CurrentDeviceResourceRestore { previous_raw };
    let device = DeviceId::current().unwrap();
    let upstream = DeviceMemoryResource::cuda_on(device).unwrap();
    let limiting = upstream.limiting(1).unwrap();
    let callback: Arc<dyn Fn(usize) + Send + Sync + 'static> = Arc::new(|_| {});
    let weak_callback: Weak<dyn Fn(usize) + Send + Sync + 'static> = Arc::downgrade(&callback);
    let callback_mr = limiting.with_failure_callback(callback).unwrap();
    {
        let borrowed = callback_mr.as_borrowed_sys();
        set_current_device_resource(&borrowed).expect("failed to install callback memory resource");
    }
    drop(callback_mr);

    assert!(
        weak_callback.upgrade().is_some(),
        "callback state must not be freed when native destroy is rejected while the resource is current"
    );
}

#[test]
fn derived_resource_retains_failure_callback_state_after_parent_handle_drop() {
    let device = DeviceId::current().unwrap();
    let upstream = DeviceMemoryResource::cuda_on(device).unwrap();
    let callback: Arc<dyn Fn(usize) + Send + Sync + 'static> = Arc::new(|_| {});
    let weak_callback: Weak<dyn Fn(usize) + Send + Sync + 'static> = Arc::downgrade(&callback);
    let callback_mr = upstream.with_failure_callback(callback).unwrap();
    let derived = callback_mr.limiting(1).unwrap();

    drop(callback_mr);

    assert!(
        weak_callback.upgrade().is_some(),
        "derived resources must retain Rust callback state while native upstream callback adaptors remain reachable"
    );

    drop(derived);
    assert!(weak_callback.upgrade().is_none(), "callback state should release when the derived resource drops");
}
