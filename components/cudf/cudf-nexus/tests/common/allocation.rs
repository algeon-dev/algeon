use std::{ffi::c_void, ptr, sync::Mutex};

use cudf::testing::prelude::*;

/// Serializes test-only swaps of the process-global current device resource.
static CURRENT_DEVICE_RESOURCE_LOCK: Mutex<()> = Mutex::new(());

fn current_device_resource_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read current device resource");
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

/// Measure the allocation-count delta while a stats memory resource is installed.
///
/// The measured closure must perform any required host-visible synchronization
/// before returning. This helper only manages the process-global allocator swap;
/// it does not synchronize CUDA work on behalf of the caller.
pub fn allocation_count_delta<T>(op: impl FnOnce() -> T) -> i64 {
    let _guard = CURRENT_DEVICE_RESOURCE_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous_raw = current_device_resource_raw();
    let stats = cudf::testing::core::stats_memory_resource().expect("failed to create stats memory resource");
    let _restore = CurrentDeviceResourceRestore { previous_raw };
    let borrowed = stats.as_borrowed_sys();

    cudf_sys::test_set_current_device_resource(&borrowed).expect("failed to install stats memory resource");

    let before = stats.testing_allocation_count().expect("failed to read allocation count before operation");
    let _result = op();
    let after = stats.testing_allocation_count().expect("failed to read allocation count after operation");
    after - before
}
