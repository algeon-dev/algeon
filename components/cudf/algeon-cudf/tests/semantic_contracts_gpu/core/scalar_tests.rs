use cudf::{
    DataType, ErrorCode, ErrorKind, ScalarValue, TimestampResolution, TypeId,
    testing::core::trigger_negative_string_scalar_length_for_test,
};

use crate::common::{error_context_value, run_named_cases, scalar, with_test_runtime};

fn test_string_scalar_value_round_trip() {
    let scalar = scalar("hello").unwrap();
    assert_eq!(scalar.string_value().unwrap(), "hello");
}

fn test_empty_string_scalar_value_round_trip() {
    let scalar = scalar("").unwrap();
    assert_eq!(scalar.string_value().unwrap(), "");
}

fn test_i32_scalar_round_trip() {
    let s = scalar(42i32).unwrap();
    assert_eq!(s.value::<i32>().unwrap(), 42);
}

fn test_i64_scalar_round_trip() {
    let s = scalar(123i64).unwrap();
    assert_eq!(s.value::<i64>().unwrap(), 123);
}

fn test_f32_scalar_round_trip() {
    let s = scalar(3.25f32).unwrap();
    let v = s.value::<f32>().unwrap();
    assert!((v - 3.25f32).abs() < 1e-6);
}

fn test_f64_scalar_round_trip() {
    let s = scalar(2.75f64).unwrap();
    let v = s.value::<f64>().unwrap();
    assert!((v - 2.75f64).abs() < 1e-12);
}

fn test_null_numeric_is_not_valid() {
    let s = scalar(ScalarValue::Null(TypeId::Int32.into())).unwrap();
    assert!(!s.is_valid().unwrap());
}

fn test_null_string_is_not_valid() {
    let s = scalar(ScalarValue::Null(TypeId::String.into())).unwrap();
    assert!(!s.is_valid().unwrap());
}

fn test_valid_numeric_is_valid() {
    let s = scalar(42i32).unwrap();
    assert!(s.is_valid().unwrap());
}

fn test_valid_string_is_valid() {
    let s = scalar("hello").unwrap();
    assert!(s.is_valid().unwrap());
}

fn test_numeric_scalar_data_type() {
    let s = scalar(42i32).unwrap();
    assert_eq!(s.data_type().unwrap().id(), TypeId::Int32);
}

fn test_string_scalar_data_type() {
    let s = scalar("x").unwrap();
    assert_eq!(s.data_type().unwrap().id(), TypeId::String);
}

fn test_decimal32_scalar_round_trip_dtype_scale_validity_and_value() {
    let s = scalar(ScalarValue::Decimal32 { raw: 12345, scale: -2 }).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal32, -2).unwrap());
    assert!(s.is_valid().unwrap());
    assert_eq!(s.decimal32_raw_value().unwrap(), 12345);
}

fn test_decimal64_scalar_round_trip_dtype_scale_validity_and_value() {
    let s = scalar(ScalarValue::Decimal64 { raw: -987654321, scale: -4 }).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4).unwrap());
    assert!(s.is_valid().unwrap());
    assert_eq!(s.decimal64_raw_value().unwrap(), -987654321);
}

fn test_decimal128_scalar_round_trip_dtype_scale_validity_and_value() {
    let raw = 123456789012345678901234567890i128;
    let s = scalar(ScalarValue::Decimal128 { raw, scale: -9 }).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal128, -9).unwrap());
    assert!(s.is_valid().unwrap());
    assert_eq!(s.decimal128_raw_value().unwrap(), raw);
}

fn test_timestamp_scalar_round_trip_dtype_validity_and_value() {
    let s =
        scalar(ScalarValue::Timestamp { raw: 1_700_000_000_123_456, resolution: TimestampResolution::Microseconds })
            .unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::new(TypeId::TimestampMicroseconds));
    assert!(s.is_valid().unwrap());
    assert_eq!(s.timestamp_raw_value().unwrap(), 1_700_000_000_123_456);
}

fn test_timestamp_days_raw_scalar_round_trip_sign_extends_i32_value() {
    let s = scalar(ScalarValue::Timestamp { raw: -12, resolution: TimestampResolution::Days }).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::new(TypeId::TimestampDays));
    assert!(s.is_valid().unwrap());
    assert_eq!(s.timestamp_raw_value().unwrap(), -12);
}

fn test_duration_scalar_round_trip_dtype_validity_and_value() {
    let s = scalar(ScalarValue::Duration { raw: -42_000, resolution: TimestampResolution::Milliseconds }).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::new(TypeId::DurationMilliseconds));
    assert!(s.is_valid().unwrap());
    assert_eq!(s.duration_raw_value().unwrap(), -42_000);
}

fn test_duration_days_raw_scalar_round_trip_sign_extends_i32_value() {
    let s = scalar(ScalarValue::Duration { raw: -7, resolution: TimestampResolution::Days }).unwrap();
    assert_eq!(s.data_type().unwrap(), DataType::new(TypeId::DurationDays));
    assert!(s.is_valid().unwrap());
    assert_eq!(s.duration_raw_value().unwrap(), -7);
}

fn test_null_raw_decimal_timestamp_and_duration_scalars_are_not_valid() {
    with_test_runtime(|rt| {
        let decimal32 = rt.scalar(ScalarValue::Null(DataType::fixed_point(TypeId::Decimal32, -2).unwrap())).unwrap();
        assert_eq!(decimal32.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal32, -2).unwrap());
        assert!(!decimal32.is_valid().unwrap());

        let decimal64 = rt.scalar(ScalarValue::Null(DataType::fixed_point(TypeId::Decimal64, -4).unwrap())).unwrap();
        assert_eq!(decimal64.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4).unwrap());
        assert!(!decimal64.is_valid().unwrap());

        let decimal128 = rt.scalar(ScalarValue::Null(DataType::fixed_point(TypeId::Decimal128, -5).unwrap())).unwrap();
        assert_eq!(decimal128.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal128, -5).unwrap());
        assert!(!decimal128.is_valid().unwrap());

        let timestamp = rt.scalar(ScalarValue::Null(TimestampResolution::Nanoseconds.type_id().into())).unwrap();
        assert_eq!(timestamp.data_type().unwrap(), DataType::new(TypeId::TimestampNanoseconds));
        assert!(!timestamp.is_valid().unwrap());

        let duration = rt.scalar(ScalarValue::Null(TimestampResolution::Seconds.duration_type_id().into())).unwrap();
        assert_eq!(duration.data_type().unwrap(), DataType::new(TypeId::DurationSeconds));
        assert!(!duration.is_valid().unwrap());
    });
}

fn test_decimal_scalar_value_rejects_non_decimal_dtype() {
    let decimal_err = ScalarValue::decimal(1, TypeId::Int64.into()).unwrap_err();
    assert_eq!(decimal_err.kind(), ErrorKind::Unsupported);
    assert_eq!(decimal_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(decimal_err.operation(), "scalar_value::decimal");
    assert_eq!(error_context_value(&decimal_err, "type_id").as_deref(), Some("int64"));
    assert_eq!(error_context_value(&decimal_err, "reason").as_deref(), Some("unsupported_scalar_dtype"));
    assert_eq!(error_context_value(&decimal_err, "expected_family").as_deref(), Some("decimal"));
}

fn test_decimal_scalar_value_rejects_raw_outside_storage_width() {
    let err = ScalarValue::decimal(i128::from(i32::MAX) + 1, DataType::fixed_point(TypeId::Decimal32, -2).unwrap())
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "scalar_value::decimal");
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("decimal32"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("raw_value_out_of_range"));
}

fn test_temporal_days_raw_constructor_validates_physical_range_before_ffi() {
    with_test_runtime(|rt| {
        let err = rt
            .scalar(ScalarValue::Timestamp { raw: i64::from(i32::MAX) + 1, resolution: TimestampResolution::Days })
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
        assert_eq!(err.operation(), "scalar::timestamp_from_raw_in_domain");
        assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("timestamp_days"));
        assert_eq!(error_context_value(&err, "reason").as_deref(), Some("raw_value_out_of_range"));

        let err = rt
            .scalar(ScalarValue::Duration { raw: i64::from(i32::MIN) - 1, resolution: TimestampResolution::Days })
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
        assert_eq!(err.operation(), "scalar::duration_from_raw_in_domain");
        assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("duration_days"));
        assert_eq!(error_context_value(&err, "reason").as_deref(), Some("raw_value_out_of_range"));
    });
}

fn test_string_scalar_rejects_interior_null_bytes_with_stable_code() {
    let err = with_test_runtime(|rt| rt.scalar("a\0b")).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "scalar::from_str_in_domain");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("string_scalar"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("interior_null_byte"));
}

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

fn test_string_scalar_numeric_value_fails() {
    let s = scalar("x").unwrap();
    let err = s.value::<i32>().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("string"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "actual_type_id").as_deref(), None);
    assert_eq!(error_context_value(&err, "requested_type_id").as_deref(), None);
}

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

    let decimal32 = scalar(ScalarValue::Decimal32 { raw: 123, scale: -2 }).unwrap();
    let err = decimal32.decimal64_raw_value().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "scalar::decimal64_raw_value");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("decimal32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("decimal64"));
}

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

fn test_string_scalar_negative_length_uses_stable_foreign_contract() {
    let err = trigger_negative_string_scalar_length_for_test().unwrap_err();

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

#[test]
fn scalar_tests_cases() {
    run_named_cases(&[
        ("test_string_scalar_value_round_trip", test_string_scalar_value_round_trip),
        ("test_empty_string_scalar_value_round_trip", test_empty_string_scalar_value_round_trip),
        ("test_i32_scalar_round_trip", test_i32_scalar_round_trip),
        ("test_i64_scalar_round_trip", test_i64_scalar_round_trip),
        ("test_f32_scalar_round_trip", test_f32_scalar_round_trip),
        ("test_f64_scalar_round_trip", test_f64_scalar_round_trip),
        ("test_null_numeric_is_not_valid", test_null_numeric_is_not_valid),
        ("test_null_string_is_not_valid", test_null_string_is_not_valid),
        ("test_valid_numeric_is_valid", test_valid_numeric_is_valid),
        ("test_valid_string_is_valid", test_valid_string_is_valid),
        ("test_numeric_scalar_data_type", test_numeric_scalar_data_type),
        ("test_string_scalar_data_type", test_string_scalar_data_type),
        (
            "test_decimal32_scalar_round_trip_dtype_scale_validity_and_value",
            test_decimal32_scalar_round_trip_dtype_scale_validity_and_value,
        ),
        (
            "test_decimal64_scalar_round_trip_dtype_scale_validity_and_value",
            test_decimal64_scalar_round_trip_dtype_scale_validity_and_value,
        ),
        (
            "test_decimal128_scalar_round_trip_dtype_scale_validity_and_value",
            test_decimal128_scalar_round_trip_dtype_scale_validity_and_value,
        ),
        (
            "test_timestamp_scalar_round_trip_dtype_validity_and_value",
            test_timestamp_scalar_round_trip_dtype_validity_and_value,
        ),
        (
            "test_timestamp_days_raw_scalar_round_trip_sign_extends_i32_value",
            test_timestamp_days_raw_scalar_round_trip_sign_extends_i32_value,
        ),
        (
            "test_duration_scalar_round_trip_dtype_validity_and_value",
            test_duration_scalar_round_trip_dtype_validity_and_value,
        ),
        (
            "test_duration_days_raw_scalar_round_trip_sign_extends_i32_value",
            test_duration_days_raw_scalar_round_trip_sign_extends_i32_value,
        ),
        (
            "test_null_raw_decimal_timestamp_and_duration_scalars_are_not_valid",
            test_null_raw_decimal_timestamp_and_duration_scalars_are_not_valid,
        ),
        ("test_decimal_scalar_value_rejects_non_decimal_dtype", test_decimal_scalar_value_rejects_non_decimal_dtype),
        (
            "test_decimal_scalar_value_rejects_raw_outside_storage_width",
            test_decimal_scalar_value_rejects_raw_outside_storage_width,
        ),
        (
            "test_temporal_days_raw_constructor_validates_physical_range_before_ffi",
            test_temporal_days_raw_constructor_validates_physical_range_before_ffi,
        ),
        (
            "test_string_scalar_rejects_interior_null_bytes_with_stable_code",
            test_string_scalar_rejects_interior_null_bytes_with_stable_code,
        ),
        ("test_type_mismatch_value_extraction", test_type_mismatch_value_extraction),
        ("test_string_scalar_numeric_value_fails", test_string_scalar_numeric_value_fails),
        (
            "test_numeric_scalar_string_value_fails_with_type_mismatch_contract",
            test_numeric_scalar_string_value_fails_with_type_mismatch_contract,
        ),
        (
            "test_decimal_raw_value_wrong_type_uses_type_mismatch_contract",
            test_decimal_raw_value_wrong_type_uses_type_mismatch_contract,
        ),
        (
            "test_temporal_raw_value_wrong_type_uses_type_mismatch_contract",
            test_temporal_raw_value_wrong_type_uses_type_mismatch_contract,
        ),
        (
            "test_string_scalar_negative_length_uses_stable_foreign_contract",
            test_string_scalar_negative_length_uses_stable_foreign_contract,
        ),
    ]);
}
