use std::{ffi::c_void, ptr, sync::Mutex};

use cudf::{DeviceId, core::DeviceMemoryResource};
use cudf_sys::test_support;

/// Serializes test-only swaps of the process-global current device resource.
static CURRENT_DEVICE_RESOURCE_LOCK: Mutex<()> = Mutex::new(());

fn current_device_resource_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let status = unsafe { test_support::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read current device resource");
    out
}

fn set_current_device_resource(mr: &cudf_sys::BorrowedMemoryResource<'_>) -> Result<(), cudf_sys::cudf_status_t> {
    let status = unsafe { test_support::cudf_test_memory_resource_set_current(mr.as_raw()) };
    if status == cudf_sys::CUDF_SUCCESS { Ok(()) } else { Err(status) }
}

unsafe fn set_current_device_resource_raw(raw_mr: *mut c_void) -> Result<(), cudf_sys::cudf_status_t> {
    let status = unsafe { test_support::cudf_test_memory_resource_set_current_raw(raw_mr) };
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

fn stats_memory_resource() -> DeviceMemoryResource {
    let device = DeviceId::current().expect("failed to resolve device for stats memory resource");
    let mut raw = ptr::null_mut();
    let status = unsafe { test_support::cudf_test_memory_resource_create_stats(&mut raw) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create stats memory resource");
    // SAFETY: the test hook returned a fresh, uniquely owned handle for the
    // current device, and this test transfers its sole ownership to cudf.
    unsafe { cudf::testing::core::adopt_owned_memory_resource_for_test(raw, device) }
        .expect("failed to adopt stats memory resource")
}

fn allocation_count(resource: &DeviceMemoryResource) -> i64 {
    let mut out = 0;
    let status = unsafe {
        test_support::cudf_test_memory_resource_allocation_count(resource.as_borrowed_sys().as_raw(), &mut out)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read allocation count");
    out
}

/// Measure the allocation-count delta while a stats memory resource is installed.
///
/// The measured closure must perform any required host-visible synchronization
/// before returning. This helper only manages the process-global allocator swap;
/// it does not synchronize CUDA work on behalf of the caller.
pub fn allocation_count_delta<T>(op: impl FnOnce() -> T) -> i64 {
    let _guard = CURRENT_DEVICE_RESOURCE_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous_raw = current_device_resource_raw();
    let stats = stats_memory_resource();
    let _restore = CurrentDeviceResourceRestore { previous_raw };
    let borrowed = stats.as_borrowed_sys();

    set_current_device_resource(&borrowed).expect("failed to install stats memory resource");

    let before = allocation_count(&stats);
    let _result = op();
    let after = allocation_count(&stats);
    after - before
}
