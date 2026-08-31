use cudf::{DataType, ErrorCode, ErrorKind, TimestampResolution, TypeId};

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
fn test_decimal32_raw_scalar_round_trip_dtype_scale_validity_and_value() {
    let s = crate::common::decimal32_raw_scalar(12345, -2).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal32, -2).unwrap());
    assert!(s.is_valid().unwrap());
    assert_eq!(s.decimal32_raw_value().unwrap(), 12345);
}

#[test]
fn test_decimal64_raw_scalar_round_trip_dtype_scale_validity_and_value() {
    let s = crate::common::decimal64_raw_scalar(-987654321, -4).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4).unwrap());
    assert!(s.is_valid().unwrap());
    assert_eq!(s.decimal64_raw_value().unwrap(), -987654321);
}

#[test]
fn test_decimal128_raw_scalar_round_trip_dtype_scale_validity_and_value() {
    let raw = 123456789012345678901234567890i128;
    let s = crate::common::decimal128_raw_scalar(raw, -9).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal128, -9).unwrap());
    assert!(s.is_valid().unwrap());
    assert_eq!(s.decimal128_raw_value().unwrap(), raw);
}

#[test]
fn test_timestamp_raw_scalar_round_trip_dtype_validity_and_value() {
    let s = crate::common::timestamp_raw_scalar(1_700_000_000_123_456, TimestampResolution::Microseconds).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::new(TypeId::TimestampMicroseconds));
    assert!(s.is_valid().unwrap());
    assert_eq!(s.timestamp_raw_value().unwrap(), 1_700_000_000_123_456);
}

#[test]
fn test_timestamp_days_raw_scalar_round_trip_sign_extends_i32_value() {
    let s = crate::common::timestamp_raw_scalar(-12, TimestampResolution::Days).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::new(TypeId::TimestampDays));
    assert!(s.is_valid().unwrap());
    assert_eq!(s.timestamp_raw_value().unwrap(), -12);
}

#[test]
fn test_duration_raw_scalar_round_trip_dtype_validity_and_value() {
    let s = crate::common::duration_raw_scalar(-42_000, TimestampResolution::Milliseconds).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::new(TypeId::DurationMilliseconds));
    assert!(s.is_valid().unwrap());
    assert_eq!(s.duration_raw_value().unwrap(), -42_000);
}

#[test]
fn test_duration_days_raw_scalar_round_trip_sign_extends_i32_value() {
    let s = crate::common::duration_raw_scalar(-7, TimestampResolution::Days).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::new(TypeId::DurationDays));
    assert!(s.is_valid().unwrap());
    assert_eq!(s.duration_raw_value().unwrap(), -7);
}

#[test]
fn test_null_raw_decimal_timestamp_and_duration_scalars_are_not_valid() {
    crate::common::with_test_runtime(|rt| {
        let decimal32 = rt.null_decimal32_raw_scalar(-2).unwrap();
        assert_eq!(decimal32.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal32, -2).unwrap());
        assert!(!decimal32.is_valid().unwrap());

        let decimal64 = rt.null_decimal64_raw_scalar(-4).unwrap();
        assert_eq!(decimal64.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4).unwrap());
        assert!(!decimal64.is_valid().unwrap());

        let decimal128 = rt.null_decimal128_raw_scalar(-5).unwrap();
        assert_eq!(decimal128.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal128, -5).unwrap());
        assert!(!decimal128.is_valid().unwrap());

        let timestamp = rt.null_timestamp_raw_scalar(TimestampResolution::Nanoseconds).unwrap();
        assert_eq!(timestamp.data_type().unwrap(), DataType::new(TypeId::TimestampNanoseconds));
        assert!(!timestamp.is_valid().unwrap());

        let duration = rt.null_duration_raw_scalar(TimestampResolution::Seconds).unwrap();
        assert_eq!(duration.data_type().unwrap(), DataType::new(TypeId::DurationSeconds));
        assert!(!duration.is_valid().unwrap());
    });
}

#[test]
fn test_raw_scalar_constructors_validate_type_families_before_ffi() {
    crate::common::with_test_runtime(|rt| {
        let decimal_err = rt.decimal_raw_scalar(1, TypeId::Int64).unwrap_err();
        assert_eq!(decimal_err.kind(), ErrorKind::Unsupported);
        assert_eq!(decimal_err.code(), ErrorCode::UnsupportedDtype);
        assert_eq!(decimal_err.operation(), "scalar::decimal_from_raw_in_domain");
        assert_eq!(error_context_value(&decimal_err, "type_id").as_deref(), Some("int64"));
        assert_eq!(error_context_value(&decimal_err, "reason").as_deref(), Some("unsupported_scalar_dtype"));
        assert_eq!(error_context_value(&decimal_err, "expected_family").as_deref(), Some("decimal"));
    });
}

#[test]
fn test_decimal_raw_constructor_validates_physical_range_before_ffi() {
    crate::common::with_test_runtime(|rt| {
        let err = rt
            .decimal_raw_scalar(i128::from(i32::MAX) + 1, DataType::fixed_point(TypeId::Decimal32, -2).unwrap())
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
        assert_eq!(err.operation(), "scalar::decimal_from_raw_in_domain");
        assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("decimal32"));
        assert_eq!(error_context_value(&err, "reason").as_deref(), Some("raw_value_out_of_range"));
    });
}

#[test]
fn test_temporal_days_raw_constructor_validates_physical_range_before_ffi() {
    crate::common::with_test_runtime(|rt| {
        let err = rt.timestamp_raw_scalar(i64::from(i32::MAX) + 1, TimestampResolution::Days).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
        assert_eq!(err.operation(), "scalar::timestamp_from_raw_in_domain");
        assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("timestamp_days"));
        assert_eq!(error_context_value(&err, "reason").as_deref(), Some("raw_value_out_of_range"));

        let err = rt.duration_raw_scalar(i64::from(i32::MIN) - 1, TimestampResolution::Days).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
        assert_eq!(err.operation(), "scalar::duration_from_raw_in_domain");
        assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("duration_days"));
        assert_eq!(error_context_value(&err, "reason").as_deref(), Some("raw_value_out_of_range"));
    });
}

#[test]
fn test_string_scalar_rejects_interior_null_bytes_with_stable_code() {
    let err = crate::common::with_test_runtime(|rt| rt.string_scalar("a\0b")).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "scalar::from_str_in_domain");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("string_scalar"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("interior_null_byte"));
}

#[test]
fn test_type_mismatch_value_extraction() {
    let s = scalar(42i32).unwrap();
    let err = s.value::<f64>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("float64"));
    assert_eq!(error_context_value(&err, "actual_type_id").as_deref(), None);
    assert_eq!(error_context_value(&err, "requested_type_id").as_deref(), None);
}

#[test]
fn test_string_scalar_numeric_value_fails() {
    let s = crate::common::string_scalar("x").unwrap();
    let err = s.value::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("string"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "actual_type_id").as_deref(), None);
    assert_eq!(error_context_value(&err, "requested_type_id").as_deref(), None);
}

#[test]
fn test_numeric_scalar_string_value_fails_with_type_mismatch_contract() {
    let s = scalar(42i32).unwrap();
    let err = s.string_value().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::string_value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("string"));
    assert_eq!(error_context_value(&err, "actual_type_id").as_deref(), None);
    assert_eq!(error_context_value(&err, "requested_type_id").as_deref(), None);
}

#[test]
fn test_decimal_raw_value_wrong_type_uses_type_mismatch_contract() {
    let int = scalar(42i32).unwrap();
    let err = int.decimal32_raw_value().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::decimal32_raw_value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("decimal32"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), None);
    assert_eq!(error_context_value(&err, "expected_family").as_deref(), None);

    let decimal32 = crate::common::decimal32_raw_scalar(123, -2).unwrap();
    let err = decimal32.decimal64_raw_value().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::decimal64_raw_value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("decimal32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("decimal64"));
}

#[test]
fn test_temporal_raw_value_wrong_type_uses_type_mismatch_contract() {
    let int = scalar(42i32).unwrap();

    let err = int.timestamp_raw_value().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::timestamp_raw_value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "expected_family").as_deref(), Some("timestamp"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), None);

    let err = int.duration_raw_value().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::duration_raw_value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "expected_family").as_deref(), Some("duration"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), None);
}

#[test]
fn test_string_scalar_negative_length_uses_stable_foreign_contract() {
    let err = cudf::testing::core::trigger_negative_string_scalar_length_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "scalar::string_value");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("string_scalar_value"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("negative_string_length"));
    assert_eq!(error_context_value(&err, "raw_len").as_deref(), Some("-1"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign scalar backend returns a non-negative string length")
    );
}
