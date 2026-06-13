use cudf::{ErrorCode, ErrorKind, TypeId};

use crate::common::{error_context_value, scalar};

#[test]
fn test_string_scalar_value_round_trip() {
    let scalar = crate::common::string_scalar("hello").unwrap();
    assert_eq!(scalar.string_value().unwrap(), "hello");
}

#[test]
fn test_empty_string_scalar_value_round_trip() {
    let scalar = crate::common::string_scalar("").unwrap();
    assert_eq!(scalar.string_value().unwrap(), "");
}

#[test]
fn test_i32_scalar_round_trip() {
    let s = scalar(42i32).unwrap();
    assert_eq!(s.value::<i32>().unwrap(), 42);
}

#[test]
fn test_i64_scalar_round_trip() {
    let s = scalar(123i64).unwrap();
    assert_eq!(s.value::<i64>().unwrap(), 123);
}

#[test]
fn test_f32_scalar_round_trip() {
    let s = scalar(3.25f32).unwrap();
    let v = s.value::<f32>().unwrap();
    assert!((v - 3.25f32).abs() < 1e-6);
}

#[test]
fn test_f64_scalar_round_trip() {
    let s = scalar(2.75f64).unwrap();
    let v = s.value::<f64>().unwrap();
    assert!((v - 2.75f64).abs() < 1e-12);
}

#[test]
fn test_null_numeric_is_not_valid() {
    let s = crate::common::null_scalar::<i32>().unwrap();
    assert!(!s.is_valid().unwrap());
}

#[test]
fn test_null_string_is_not_valid() {
    let s = crate::common::null_string_scalar().unwrap();
    assert!(!s.is_valid().unwrap());
}

#[test]
fn test_valid_numeric_is_valid() {
    let s = scalar(42i32).unwrap();
    assert!(s.is_valid().unwrap());
}

#[test]
fn test_valid_string_is_valid() {
    let s = crate::common::string_scalar("hello").unwrap();
    assert!(s.is_valid().unwrap());
}

#[test]
fn test_numeric_scalar_data_type() {
    let s = scalar(42i32).unwrap();
    assert_eq!(s.data_type().unwrap().id(), TypeId::Int32);
}

#[test]
fn test_string_scalar_data_type() {
    let s = crate::common::string_scalar("x").unwrap();
    assert_eq!(s.data_type().unwrap().id(), TypeId::String);
}

#[test]
fn test_string_scalar_rejects_interior_null_bytes_with_stable_code() {
    let err = crate::common::with_test_runtime(|rt| rt.string_scalar("a\0b")).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::InvalidInput);
    assert_eq!(err.operation(), "scalar::from_str_in_domain");
    assert_eq!(error_context_value(&err, "resource"), Some("string_scalar"));
    assert_eq!(error_context_value(&err, "reason"), Some("interior_null_byte"));
}

#[test]
fn test_type_mismatch_value_extraction() {
    let s = scalar(42i32).unwrap();
    let err = s.value::<f64>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::value");
    assert_eq!(error_context_value(&err, "left_type_id"), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id"), Some("float64"));
    assert_eq!(error_context_value(&err, "actual_type_id"), None);
    assert_eq!(error_context_value(&err, "requested_type_id"), None);
}

#[test]
fn test_string_scalar_numeric_value_fails() {
    let s = crate::common::string_scalar("x").unwrap();
    let err = s.value::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::value");
    assert_eq!(error_context_value(&err, "left_type_id"), Some("string"));
    assert_eq!(error_context_value(&err, "right_type_id"), Some("int32"));
    assert_eq!(error_context_value(&err, "actual_type_id"), None);
    assert_eq!(error_context_value(&err, "requested_type_id"), None);
}

#[test]
fn test_numeric_scalar_string_value_fails_with_type_mismatch_contract() {
    let s = scalar(42i32).unwrap();
    let err = s.string_value().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::string_value");
    assert_eq!(error_context_value(&err, "left_type_id"), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id"), Some("string"));
    assert_eq!(error_context_value(&err, "actual_type_id"), None);
    assert_eq!(error_context_value(&err, "requested_type_id"), None);
}

#[test]
fn test_string_scalar_negative_length_uses_stable_foreign_contract() {
    let err = cudf::testing::trigger_negative_string_scalar_length_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "scalar::string_value");
    assert_eq!(error_context_value(&err, "resource"), Some("string_scalar_value"));
    assert_eq!(error_context_value(&err, "reason"), Some("negative_string_length"));
    assert_eq!(error_context_value(&err, "raw_len"), Some("-1"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some("ensure the foreign scalar backend returns a non-negative string length")
    );
}
