//! DeviceGuard integration tests — single-GPU (always run).
//!
//! DeviceGuard is `pub(crate)`, so these tests exercise it indirectly through
//! Runtime::builder().with_device() which internally uses DeviceGuard.

use cudf::{DeviceId, Runtime};

#[test]
fn runtime_build_with_device_zero_succeeds() {
    // DeviceGuard is used internally by RuntimeBuilder::build() when a device is specified.
    let rt = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    assert_eq!(rt.device().ordinal(), 0);
}

#[test]
fn runtime_build_without_device_uses_current() {
    // Without explicit device, build() snapshots the current device (0).
    let rt = Runtime::builder().build().unwrap();
    assert_eq!(rt.device().ordinal(), 0);
}

#[test]
fn multiple_runtimes_on_same_device_coexist() {
    let rt1 = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    let rt2 = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    assert_eq!(rt1.device(), rt2.device());
}
