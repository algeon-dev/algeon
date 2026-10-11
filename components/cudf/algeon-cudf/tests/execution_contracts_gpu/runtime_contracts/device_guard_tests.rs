//! DeviceGuard integration tests; single-GPU (always run).
//!
//! DeviceGuard is `pub(crate)`, so these tests exercise it indirectly through
//! Runtime::builder().with_device() which internally uses DeviceGuard.

use cudf::Runtime;

fn runtime_build_without_device_uses_current() {
    // Without explicit device, build() snapshots the current device (0).
    let rt = Runtime::builder().build().unwrap();
    assert_eq!(rt.device().ordinal(), 0);
}

pub(crate) const CASES: &[(&str, fn())] = &[(
    "runtime_contracts::device_guard_tests::runtime_build_without_device_uses_current",
    runtime_build_without_device_uses_current,
)];
