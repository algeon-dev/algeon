//! DeviceId integration tests; single-GPU (always run).

use cudf::{DeviceId, ErrorCode, ErrorKind};

use crate::common::{error_context_value, run_named_cases};

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

fn count_is_idempotent() {
    let a = DeviceId::count().unwrap();
    let b = DeviceId::count().unwrap();
    assert_eq!(a, b);
}

fn default_equals_device_zero() {
    let zero = DeviceId::new(0).unwrap();
    assert_eq!(DeviceId::DEFAULT, zero);
}

fn ordinal_round_trips() {
    let id = DeviceId::new(0).unwrap();
    assert_eq!(id.ordinal(), 0);
}

fn device_id_is_copy() {
    let id = DeviceId::new(0).unwrap();
    let copy = id;
    assert_eq!(id, copy);
}

#[test]
fn device_id_tests_cases() {
    run_named_cases(&[
        ("new_device_zero_succeeds", new_device_zero_succeeds),
        ("new_device_out_of_range_returns_error", new_device_out_of_range_returns_error),
        ("current_returns_device_zero", current_returns_device_zero),
        ("count_returns_at_least_one", count_returns_at_least_one),
        ("count_is_idempotent", count_is_idempotent),
        ("default_equals_device_zero", default_equals_device_zero),
        ("ordinal_round_trips", ordinal_round_trips),
        ("device_id_is_copy", device_id_is_copy),
    ]);
}
