//! DeviceId integration tests; single-GPU (always run).

use cudf::{DeviceId, ErrorCode, ErrorKind};

use crate::common::error_context_value;

fn new_device_zero_succeeds() {
    let id = DeviceId::new(0).unwrap();
    assert_eq!(id.ordinal(), 0);
}

fn new_device_out_of_range_returns_error() {
    let err = DeviceId::new(999).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::IndexOutOfBounds);
    assert_eq!(err.operation(), "DeviceId::new");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("cuda_device"));
    assert_eq!(error_context_value(&err, "ordinal").as_deref(), Some("999"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("device_ordinal_out_of_range"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("choose a device ordinal less than DeviceId::count()")
    );
    assert_ne!(err.kind(), ErrorKind::NotFound);
}

fn current_returns_device_zero() {
    let id = DeviceId::current().unwrap();
    assert_eq!(id.ordinal(), 0);
}

fn count_returns_at_least_one() {
    let n = DeviceId::count().unwrap();
    assert!(n >= 1, "expected at least 1 device, got {n}");
}

pub(crate) const CASES: &[(&str, fn())] = &[
    ("runtime_contracts::device_id_tests::new_device_zero_succeeds", new_device_zero_succeeds),
    (
        "runtime_contracts::device_id_tests::new_device_out_of_range_returns_error",
        new_device_out_of_range_returns_error,
    ),
    ("runtime_contracts::device_id_tests::current_returns_device_zero", current_returns_device_zero),
    ("runtime_contracts::device_id_tests::count_returns_at_least_one", count_returns_at_least_one),
];
