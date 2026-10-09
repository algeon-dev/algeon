//! Runtime device integration tests; single-GPU (always run).

use cudf::{DeviceId, Runtime};

fn runtime_device_accessor_returns_build_device() {
    let rt = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    assert_eq!(rt.device(), DeviceId::new(0).unwrap());
}

pub(crate) const CASES: &[(&str, fn())] = &[(
    "runtime_contracts::runtime_device_tests::runtime_device_accessor_returns_build_device",
    runtime_device_accessor_returns_build_device,
)];
