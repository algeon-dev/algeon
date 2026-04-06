//! DeviceId integration tests — single-GPU (always run).

use cudf::DeviceId;

#[test]
fn new_device_zero_succeeds() {
    let id = DeviceId::new(0).unwrap();
    assert_eq!(id.ordinal(), 0);
}

#[test]
fn new_device_out_of_range_returns_error() {
    let err = DeviceId::new(999).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn current_returns_device_zero() {
    let id = DeviceId::current().unwrap();
    assert_eq!(id.ordinal(), 0);
}

#[test]
fn count_returns_at_least_one() {
    let n = DeviceId::count().unwrap();
    assert!(n >= 1, "expected at least 1 device, got {n}");
}

#[test]
fn count_is_idempotent() {
    let a = DeviceId::count().unwrap();
    let b = DeviceId::count().unwrap();
    assert_eq!(a, b);
}

#[test]
fn default_equals_device_zero() {
    let zero = DeviceId::new(0).unwrap();
    assert_eq!(DeviceId::DEFAULT, zero);
}

#[test]
fn ordinal_round_trips() {
    let id = DeviceId::new(0).unwrap();
    assert_eq!(id.ordinal(), 0);
}

#[test]
fn device_id_is_copy() {
    let id = DeviceId::new(0).unwrap();
    let copy = id;
    assert_eq!(id, copy);
}
