//! Specification tests for type casting (`astype` / `cast`).
//!
//! These tests systematically cover cast behavior across type pairs,
//! including edge cases that are not covered by the basic unary tests.
//!
//! All tests require a GPU.

use std::sync::Arc;

use arrow::{
    array::{Array, Decimal128Array},
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, DataType, ErrorCode, ErrorKind, Runtime, TimestampResolution, TypeId,
    core::error::{FactKeyName, FactValue},
};

fn decimal128_series_from_arrow(values: Vec<Option<i128>>, precision: u8, scale: i8) -> cudf::Series {
    let amount = Decimal128Array::from(values).with_precision_and_scale(precision, scale).unwrap();
    let schema = Schema::new(vec![Field::new("amount", ArrowDataType::Decimal128(precision, scale), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(amount)]).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    let df = DataFrame::from_record_batch(&runtime, &batch).unwrap();
    df.series("amount").unwrap()
}

// ===========================================================================
// Integer width widening (lossless)
// ===========================================================================

fn test_cast_i8_to_i16() {
    let s = crate::common::series::<i8>("x", &[i8::MIN, 0, i8::MAX]).unwrap();
    let result = s.astype(TypeId::Int16).unwrap();
    let vals = result.to_vec_strict::<i16>().unwrap();
    assert_eq!(vals, vec![i8::MIN as i16, 0, i8::MAX as i16]);
}

fn test_cast_i16_to_i32() {
    let s = crate::common::series::<i16>("x", &[i16::MIN, 0, i16::MAX]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![i16::MIN as i32, 0, i16::MAX as i32]);
}

fn test_cast_i32_to_i64() {
    let s = crate::common::series::<i32>("x", &[i32::MIN, 0, i32::MAX]).unwrap();
    let result = s.astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![i32::MIN as i64, 0, i32::MAX as i64]);
}

// ===========================================================================
// Float ↔ Integer
// ===========================================================================

fn test_cast_i32_to_f32() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.astype(TypeId::Float32).unwrap();
    let vals = result.to_vec_strict::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

fn test_cast_f32_to_i32_truncates() {
    let s = crate::common::series::<f32>("x", &[1.9, -2.7, 0.1]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, -2, 0]);
}

fn test_cast_f64_to_f32() {
    let s = crate::common::series::<f64>("x", &[1.5, 2.5, 3.5]).unwrap();
    let result = s.astype(TypeId::Float32).unwrap();
    let vals = result.to_vec_strict::<f32>().unwrap();
    assert_eq!(vals, vec![1.5, 2.5, 3.5]);
}

fn test_cast_decimal128_to_f64_uses_decimal_scale_and_preserves_nulls() {
    let amount = decimal128_series_from_arrow(vec![Some(12345), None, Some(-678)], 20, 2);

    let result = amount.astype(TypeId::Float64).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.null_count(), 1);
    assert_eq!(result.to_optional_vec::<f64>().unwrap(), vec![Some(123.45), None, Some(-6.78)]);
}

fn test_cast_decimal64_to_f64_uses_decimal_scale() {
    let amount = decimal128_series_from_arrow(vec![Some(12345), Some(-678)], 20, 2)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();

    let result = amount.astype(TypeId::Float64).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![123.45, -6.78]);
}

fn test_cast_f64_to_decimal128_uses_target_scale_and_preserves_nulls() {
    let s = crate::common::nullable_series::<f64>("x", &[123.45, 0.0, -6.78], &[true, false, true]).unwrap();

    let result = s.astype(DataType::fixed_point(TypeId::Decimal128, -2).unwrap()).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal128, -2).unwrap());
    assert_eq!(result.null_count(), 1);
    let batch = DataFrame::from_series([("x", result)]).unwrap().to_record_batch().unwrap();
    let values = batch.column(0).as_any().downcast_ref::<Decimal128Array>().unwrap();
    assert_eq!(values.values().as_ref(), &[12345, 0, -678]);
    assert!(values.is_valid(0));
    assert!(values.is_null(1));
    assert!(values.is_valid(2));
}

// ===========================================================================
// Bool ↔ Integer
// ===========================================================================

fn test_cast_bool_to_i32() {
    let s = crate::common::series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 0, 1]);
}

fn test_cast_i32_to_bool() {
    let s = crate::common::series::<i32>("x", &[0, 1, 42, -1]).unwrap();
    let result = s.astype(TypeId::Bool8).unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![false, true, true, true]);
}

fn test_cast_bool_to_i64() {
    let s = crate::common::series::<bool>("x", &[false, true]).unwrap();
    let result = s.astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![0, 1]);
}

// ===========================================================================
// Unsigned ↔ Signed
// ===========================================================================

fn test_cast_u8_to_i16() {
    let s = crate::common::series::<u8>("x", &[0, 127, 255]).unwrap();
    let result = s.astype(TypeId::Int16).unwrap();
    let vals = result.to_vec_strict::<i16>().unwrap();
    assert_eq!(vals, vec![0, 127, 255]);
}

fn test_cast_u32_to_i64() {
    let s = crate::common::series::<u32>("x", &[0, u32::MAX]).unwrap();
    let result = s.astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![0, u32::MAX as i64]);
}

// ===========================================================================
// Null preservation through cast
// ===========================================================================

fn test_cast_preserves_null_mask() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    assert_eq!(result.null_count(), 1);
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(10.0), None, Some(30.0)]);
}

fn test_cast_all_null_column() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    assert_eq!(result.null_count(), 2);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![None, None]);
}

// ===========================================================================
// Empty column cast
// ===========================================================================

fn test_cast_empty_column() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    assert!(result.is_empty());
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
}

// ===========================================================================
// Same-type cast (noop)
// ===========================================================================

fn test_cast_same_type() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// Series display omits schema names
// ===========================================================================

fn test_cast_display_omits_input_name() {
    let s = crate::common::series::<i32>("my_column", &[1, 2]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("my_column"), "cast output should not expose schema names: {}", output);
}

// ===========================================================================
// is_supported_cast
// ===========================================================================

fn decimal_rescale_workspace_contract_is_scale_sensitive() {
    let scale_zero = DataType::fixed_point(TypeId::Decimal128, 0).unwrap();
    let scale_two = DataType::fixed_point(TypeId::Decimal128, -2).unwrap();

    assert_eq!(
        cudf::ops::unary::decimal_rescale_temporary_workspace_reservation_bytes(&scale_zero, &scale_zero),
        Some(0)
    );
    assert_eq!(
        cudf::ops::unary::decimal_rescale_temporary_workspace_reservation_bytes(&scale_zero, &scale_two),
        Some(256)
    );
    assert_eq!(
        cudf::ops::unary::decimal_rescale_temporary_workspace_reservation_bytes(
            &DataType::fixed_point(TypeId::Decimal64, 0).unwrap(),
            &scale_two,
        ),
        None
    );
}

fn test_is_supported_cast_numeric_to_float() {
    let from = DataType::new(TypeId::Int32);
    let to = DataType::new(TypeId::Float64);
    assert!(cudf::ops::unary::is_supported_cast(&from, &to).unwrap());
}

fn test_is_supported_cast_decimal_to_float() {
    let to = DataType::new(TypeId::Float64);
    assert!(cudf::ops::unary::is_supported_cast(&DataType::fixed_point(TypeId::Decimal64, -2).unwrap(), &to).unwrap());
    assert!(cudf::ops::unary::is_supported_cast(&DataType::fixed_point(TypeId::Decimal128, -2).unwrap(), &to).unwrap());
}

fn test_is_supported_cast_float_to_decimal() {
    let from = DataType::new(TypeId::Float64);
    assert!(
        cudf::ops::unary::is_supported_cast(&from, &DataType::fixed_point(TypeId::Decimal128, -2).unwrap()).unwrap()
    );
}

fn test_is_supported_cast_float_to_int() {
    let from = DataType::new(TypeId::Float64);
    let to = DataType::new(TypeId::Int32);
    assert!(cudf::ops::unary::is_supported_cast(&from, &to).unwrap());
}

fn test_is_supported_cast_bool_to_int() {
    let from = DataType::new(TypeId::Bool8);
    let to = DataType::new(TypeId::Int32);
    assert!(cudf::ops::unary::is_supported_cast(&from, &to).unwrap());
}

// ===========================================================================
// Integer-to-timestamp cast (supported by libcudf)
// ===========================================================================

fn test_cast_i64_to_timestamp_microseconds() {
    // libcudf requires int64 → duration → timestamp (direct int→timestamp is not supported).
    let s = crate::common::series::<i64>("x", &[0, 1_000_000, 1_618_000_000_000_000]).unwrap();
    let dur = s.astype(TypeId::DurationMicroseconds).unwrap();
    let result = dur.astype(TypeId::TimestampMicroseconds).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
    assert_eq!(result.len(), 3);
}

fn test_cast_timestamp_to_i64() {
    // Round-trip: i64 -> duration_us -> timestamp_us -> duration_us -> i64
    let s = crate::common::series::<i64>("x", &[0, 1000, 2000]).unwrap();
    let dur = s.astype(TypeId::DurationMicroseconds).unwrap();
    let ts = dur.astype(TypeId::TimestampMicroseconds).unwrap();
    let back_dur = ts.astype(TypeId::DurationMicroseconds).unwrap();
    let back = back_dur.astype(TypeId::Int64).unwrap();
    assert_eq!(back.to_vec_strict::<i64>().unwrap(), vec![0, 1000, 2000]);
}

// ===========================================================================
// Convenience method: to_datetime
// ===========================================================================

fn test_to_datetime_default_microseconds() {
    let s = crate::common::series::<i64>("ts", &[0, 1_000_000]).unwrap();
    let result = s.to_datetime(None).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
}

fn test_to_datetime_explicit_milliseconds() {
    let s = crate::common::series::<i64>("ts", &[0, 1000]).unwrap();
    let result = s.to_datetime(Some(TimestampResolution::Milliseconds)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMilliseconds);
}

fn test_to_datetime_nanoseconds() {
    let s = crate::common::series::<i64>("ts", &[0, 1_000_000_000]).unwrap();
    let result = s.to_datetime(Some(TimestampResolution::Nanoseconds)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampNanoseconds);
}

fn test_to_datetime_seconds() {
    let s = crate::common::series::<i64>("ts", &[0, 86400]).unwrap();
    let result = s.to_datetime(Some(TimestampResolution::Seconds)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampSeconds);
}

fn test_to_datetime_round_trip() {
    let s = crate::common::series::<i64>("ts", &[0, 1000, 2000]).unwrap();
    let dt = s.to_datetime(Some(TimestampResolution::Microseconds)).unwrap();
    // libcudf requires timestamp → duration → int64 for the reverse path.
    let dur = dt.astype(TypeId::DurationMicroseconds).unwrap();
    let back = dur.astype(TypeId::Int64).unwrap();
    assert_eq!(back.to_vec_strict::<i64>().unwrap(), vec![0, 1000, 2000]);
}

fn test_to_datetime_preserves_nulls() {
    let s = crate::common::nullable_series::<i64>("ts", &[100, 0, 300], &[true, false, true]).unwrap();
    let result = s.to_datetime(None).unwrap();
    assert_eq!(result.null_count(), 1);
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
}

// ===========================================================================
// Convenience method: to_numeric
// ===========================================================================

fn test_to_numeric_f64_to_i64() {
    let s = crate::common::series::<f64>("x", &[1.9, 2.0, 3.1]).unwrap();
    let result = s.to_numeric(TypeId::Int64).unwrap();
    assert_eq!(result.to_vec_strict::<i64>().unwrap(), vec![1, 2, 3]);
}

fn test_to_numeric_i32_to_f64() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.to_numeric(TypeId::Float64).unwrap();
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
}

fn test_to_numeric_bool_to_i32() {
    let s = crate::common::series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.to_numeric(TypeId::Int32).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 0, 1]);
}

fn test_to_numeric_rejects_non_numeric_target() {
    let s = crate::common::series::<i32>("x", &[1, 2]).unwrap();
    let err = s.to_numeric(TypeId::String).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "series::to_numeric");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
}

fn test_to_numeric_rejects_timestamp_target() {
    let s = crate::common::series::<i32>("x", &[1, 2]).unwrap();
    let err = s.to_numeric(TypeId::TimestampMicroseconds).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "series::to_numeric");
}

#[test]
fn cast_spec_tests_cases() {
    crate::common::run_named_cases(&[
        ("test_cast_i8_to_i16", test_cast_i8_to_i16),
        ("test_cast_i16_to_i32", test_cast_i16_to_i32),
        ("test_cast_i32_to_i64", test_cast_i32_to_i64),
        ("test_cast_i32_to_f32", test_cast_i32_to_f32),
        ("test_cast_f32_to_i32_truncates", test_cast_f32_to_i32_truncates),
        ("test_cast_f64_to_f32", test_cast_f64_to_f32),
        (
            "test_cast_decimal128_to_f64_uses_decimal_scale_and_preserves_nulls",
            test_cast_decimal128_to_f64_uses_decimal_scale_and_preserves_nulls,
        ),
        ("test_cast_decimal64_to_f64_uses_decimal_scale", test_cast_decimal64_to_f64_uses_decimal_scale),
        (
            "test_cast_f64_to_decimal128_uses_target_scale_and_preserves_nulls",
            test_cast_f64_to_decimal128_uses_target_scale_and_preserves_nulls,
        ),
        ("test_cast_bool_to_i32", test_cast_bool_to_i32),
        ("test_cast_i32_to_bool", test_cast_i32_to_bool),
        ("test_cast_bool_to_i64", test_cast_bool_to_i64),
        ("test_cast_u8_to_i16", test_cast_u8_to_i16),
        ("test_cast_u32_to_i64", test_cast_u32_to_i64),
        ("test_cast_preserves_null_mask", test_cast_preserves_null_mask),
        ("test_cast_all_null_column", test_cast_all_null_column),
        ("test_cast_empty_column", test_cast_empty_column),
        ("test_cast_same_type", test_cast_same_type),
        ("test_cast_display_omits_input_name", test_cast_display_omits_input_name),
        (
            "decimal_rescale_workspace_contract_is_scale_sensitive",
            decimal_rescale_workspace_contract_is_scale_sensitive,
        ),
        ("test_is_supported_cast_numeric_to_float", test_is_supported_cast_numeric_to_float),
        ("test_is_supported_cast_decimal_to_float", test_is_supported_cast_decimal_to_float),
        ("test_is_supported_cast_float_to_decimal", test_is_supported_cast_float_to_decimal),
        ("test_is_supported_cast_float_to_int", test_is_supported_cast_float_to_int),
        ("test_is_supported_cast_bool_to_int", test_is_supported_cast_bool_to_int),
        ("test_cast_i64_to_timestamp_microseconds", test_cast_i64_to_timestamp_microseconds),
        ("test_cast_timestamp_to_i64", test_cast_timestamp_to_i64),
        ("test_to_datetime_default_microseconds", test_to_datetime_default_microseconds),
        ("test_to_datetime_explicit_milliseconds", test_to_datetime_explicit_milliseconds),
        ("test_to_datetime_nanoseconds", test_to_datetime_nanoseconds),
        ("test_to_datetime_seconds", test_to_datetime_seconds),
        ("test_to_datetime_round_trip", test_to_datetime_round_trip),
        ("test_to_datetime_preserves_nulls", test_to_datetime_preserves_nulls),
        ("test_to_numeric_f64_to_i64", test_to_numeric_f64_to_i64),
        ("test_to_numeric_i32_to_f64", test_to_numeric_i32_to_f64),
        ("test_to_numeric_bool_to_i32", test_to_numeric_bool_to_i32),
        ("test_to_numeric_rejects_non_numeric_target", test_to_numeric_rejects_non_numeric_target),
        ("test_to_numeric_rejects_timestamp_target", test_to_numeric_rejects_timestamp_target),
    ]);
}
