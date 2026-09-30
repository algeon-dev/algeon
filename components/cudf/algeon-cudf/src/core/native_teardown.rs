//! Shared fail-closed teardown for native handles pinned to one CUDA device.

use crate::runtime::{self, DeviceFaultKind, DeviceId};

/// Consume a device-owned native handle after proving owner-device entry.
///
/// `destroy` follows the cudf-sys teardown ABI: ownership is consumed on call
/// regardless of status. A non-success status is therefore diagnostic only;
/// this helper never retries or reuses the handle. Fatal CUDA status closes
/// process-lifetime admission for the device, while every native teardown
/// failure is counted as quarantined cleanup.
pub(crate) fn drop_device_owned_handle<T>(
    raw: *mut T,
    device: DeviceId,
    operation: &'static str,
    destroy: impl FnOnce(*mut T) -> cudf_sys::cudf_status_t,
) {
    let mut raw = raw;
    let mut status = None;
    runtime::drop_device_resource(&mut raw, device, operation, |owned| {
        status = Some(destroy(owned));
    });

    let Some(status) = status.filter(|status| *status != cudf_sys::CUDF_SUCCESS) else {
        return;
    };
    if status == cudf_sys::CUDF_ERROR_FATAL_CUDA {
        runtime::report_device_fault(
            device,
            DeviceFaultKind::FatalDeviceOperation,
            operation,
            Some(runtime::DeviceFaultNativeStatus::Cudf(status)),
        );
    }
    runtime::device_health::record_teardown_failure(
        device,
        operation,
        &format_args!("native teardown failed with status {status}"),
    );
}
